/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <algorithm>
#include <cassert>
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <nccl/tuner.h>

#include "internal/tuner/nccl_defaults.h"
#include "nccl_ofi_log.h"
#include "nccl_ofi_math.h"
#include "nccl_ofi_pthread.h"
#include "nccl_ofi_system.h"
#include "nccl_ofi_param.h"
#include "nccl_ofi_platform.h"

#include "tuner/nccl_ofi_tuner_region.h"
#include "tuner/nccl_ofi_tuner_model.h"
#include "tuner/nccl_ofi_tuner.h"
#include "tuner/nccl_ofi_tuner_process_config.h"

pthread_mutex_t nccl_ofi_tuner_ctx_lock = PTHREAD_MUTEX_INITIALIZER;

static ncclResult_t nccl_ofi_tuner_destroy(void *context)
{
	ncclResult_t ret = ncclSuccess;
	nccl_ofi_tuner_context_t *ctx = (nccl_ofi_tuner_context_t *)context;

	nccl_net_ofi_mutex_lock(&nccl_ofi_tuner_ctx_lock);
	if (ctx != NULL) {
		if (ctx->destroy_internal != NULL) {
			ret = ctx->destroy_internal(ctx);
		}
		free(ctx);
	}
	nccl_net_ofi_mutex_unlock(&nccl_ofi_tuner_ctx_lock);

	return ret;
}

static ncclResult_t nccl_ofi_tuner_init(size_t nRanks, size_t nNodes, ncclDebugLogger_t logFunction, void **context)
{
	*context = NULL;

	if (ofi_log_function == NULL) {
		ofi_log_function = logFunction;
	}

	/* Ensure parameters are initialized.  When the tuner is loaded as a
	   separate shared library, it gets its own copy of the parameter
	   space that the net plugin init path does not reach. */
	int param_ret = ofi_nccl_parameters_init();
	if (OFI_UNLIKELY(param_ret != 0)) {
		return ncclInternalError;
	}

	nccl_net_ofi_mutex_lock(&nccl_ofi_tuner_ctx_lock);

	// Static instance ensures one-time initialization per process
	static TunerProcessConfig constants;

	/* Check if OFI tuner should be used based on platform and environment */
	if (!constants.should_use_ofi_tuner()) {
		constants.log_fallback_reason();
		nccl_net_ofi_mutex_unlock(&nccl_ofi_tuner_ctx_lock);
		return ncclSuccess;
	}

	/* Check if platform supports region or model tuner for this configuration */
	bool region_support = is_region_supported(constants.get_tuner_platform(), nRanks, nNodes);
	bool model_support = is_model_supported(constants.get_tuner_platform(), nRanks, nNodes);

	if (!region_support && !model_support) {
		NCCL_OFI_INFO(NCCL_INIT | NCCL_TUNING,
			"NCCL_OFI_TUNER is not available for platform : %s, Fall back to NCCL's tuner",
			constants.get_platform_type());
		nccl_net_ofi_mutex_unlock(&nccl_ofi_tuner_ctx_lock);
		return ncclSuccess;
	}

	/* Allocate tuner context */
	nccl_ofi_tuner_context_t *ctx = static_cast<nccl_ofi_tuner_context_t *>(calloc(1, sizeof(nccl_ofi_tuner_context_t)));
	if (ctx == NULL) {
		NCCL_OFI_WARN("Context allocation failed.");
		nccl_net_ofi_mutex_unlock(&nccl_ofi_tuner_ctx_lock);
		return ncclInternalError;
	}

	ctx->nRanks = nRanks;
	ctx->nNodes = nNodes;

	/*
	 * Choose "Region" over "Model" when both are supported.
	 * TUNER_TYPE env variable is ignored if the forced tuner type is not
	 * supported by the given platform, nRanks and nNodes.
	 */
	if (region_support && !(model_support && constants.should_force_model_tuner())) {
		ctx->type = TUNER_TYPE::REGION;
		ctx->init_internal = region_init_internal;
		ctx->get_coll_info_internal_v6 = region_get_coll_info_internal_v6;
		ctx->get_coll_info_internal_v3 = region_get_coll_info_internal_v3;
		ctx->get_coll_info_internal_v2 = region_get_coll_info_internal_v2;
		ctx->destroy_internal = region_destroy_internal;
		NCCL_OFI_INFO(NCCL_INIT | NCCL_TUNING, "Region base Tuner is chosen for platform: %s",
			constants.get_platform_type());
	} else {
		assert(model_support);
		ctx->type = TUNER_TYPE::MODEL;
		ctx->init_internal = model_init_internal;
		ctx->get_coll_info_internal_v6 = model_get_coll_info_internal_v6;
		ctx->get_coll_info_internal_v3 = model_get_coll_info_internal_v3;
		ctx->get_coll_info_internal_v2 = model_get_coll_info_internal_v2;
		ctx->destroy_internal = model_destroy_internal;
		NCCL_OFI_INFO(NCCL_INIT | NCCL_TUNING, "Model base Tuner is chosen for platform: %s",
			constants.get_platform_type());
	}

	/* Initialize the selected tuner */
	ncclResult_t ret = ctx->init_internal(ctx, constants.get_tuner_platform(), nRanks, nNodes);

	NCCL_OFI_INFO(NCCL_INIT | NCCL_TUNING, "Tuner init: comm with %ld ranks and %ld nodes.", nRanks, nNodes);

	if (ret != ncclSuccess) {
		nccl_ofi_tuner_destroy((void *)ctx);
		ctx = NULL;
	}

	*context = (void *)ctx;
	nccl_net_ofi_mutex_unlock(&nccl_ofi_tuner_ctx_lock);

	return ret;
}


static ncclResult_t nccl_ofi_tuner_init_v2(size_t nRanks, size_t nNodes, ncclDebugLogger_t logFunction, void **context)
{
	/*
	 * NCCL parses these variables and applies user filters inside its
	 * current tuner logic. The tuner_v2 does not support setting these
	 * variables and so the internal tuner will be used instead.
	 */
	if (getenv("NCCL_ALGO") || getenv("NCCL_PROTO")) {
		NCCL_OFI_INFO(NCCL_INIT | NCCL_TUNING, "The tuner plugin can not be loaded when "
				"explicitly choosing an algorithm or protocol "
				"with NCCL_ALGO/NCCL_PROTO. "
				"Defaulting to internal tuner.");
		*context = nullptr;
		return ncclSuccess;
	}
	return nccl_ofi_tuner_init(nRanks, nNodes, logFunction, context);
}


static ncclResult_t nccl_ofi_tuner_get_coll_info(void *context,
						 ncclFunc_t collType,
						 size_t nBytes,
						 int numPipeOps,
						 float **collCostTable,
						 int numAlgo,
						 int numProto,
						 int *nChannels)
{
	ncclResult_t ret;

	nccl_ofi_tuner_context_t *ctx = (nccl_ofi_tuner_context_t *)context;
	if (ctx == NULL || ctx->get_coll_info_internal_v3 == NULL) {
		/* Fall back to NCCL's tuner */
		return ncclSuccess;
	}

	ret = ctx->get_coll_info_internal_v3(ctx, collType, nBytes, numPipeOps, collCostTable, numAlgo, numProto, nChannels);

	return ret;
}

extern "C" const ncclTuner_v3_t ncclTunerPlugin_v3 = {.name = "nccl_ofi_tuner",
					   .init = nccl_ofi_tuner_init,
					   .getCollInfo = nccl_ofi_tuner_get_coll_info,
					   .destroy = nccl_ofi_tuner_destroy};

/* **** V6 **** */
static ncclResult_t nccl_ofi_tuner_init_v6(void** ctx, uint64_t commId, size_t nRanks, size_t nNodes,
					    ncclDebugLogger_t logFunction,
					    ncclNvlDomainInfo_v6_t* nvlDomainInfo,
					    ncclTunerConstants_v6_t* constants)
{
	// if (getenv("NCCL_ALGO") || getenv("NCCL_PROTO")) {
	// 	NCCL_OFI_INFO(NCCL_INIT | NCCL_TUNING, "The tuner plugin can not be loaded when "
	// 			"explicitly choosing an algorithm or protocol "
	// 			"with NCCL_ALGO/NCCL_PROTO. "
	// 			"Defaulting to internal tuner.");
	// 	*ctx = nullptr;
	// 	return ncclSuccess;
	// }
	return nccl_ofi_tuner_init(nRanks, nNodes, logFunction, ctx);
}

static ncclResult_t nccl_ofi_tuner_get_coll_info_v6(void *context,
						     ncclFunc_t collType,
						     size_t nBytes,
						     int numPipeOps,
						     float **collCostTable,
						     int numAlgo,
						     int numProto,
						     int regBuff,
						     int *nChannels)
{
	nccl_ofi_tuner_context_t *ctx = (nccl_ofi_tuner_context_t *)context;
	if (ctx == NULL || ctx->get_coll_info_internal_v6 == NULL) {
		return ncclSuccess;
	}

	return ctx->get_coll_info_internal_v6(ctx, collType, nBytes, numPipeOps,
					      collCostTable, numAlgo, numProto, regBuff, nChannels);
}

static ncclResult_t nccl_ofi_tuner_finalize(void *context)
{
	return nccl_ofi_tuner_destroy(context);
}

static ncclResult_t nccl_ofi_tuner_get_chunk_size(void *context, ncclFunc_t collType, size_t nBytes,
						   int algo, int proto, int nChannels, size_t *chunkSize)
{
	nccl_ofi_tuner_context_t *ctx = (nccl_ofi_tuner_context_t *)context;
	if (ctx == nullptr) {
		return ncclSuccess;
	}

	if (collType == ncclFuncAllReduce && algo == NCCL_ALGO_TREE && proto == NCCL_PROTO_LL128) {
		size_t nRanks = ctx->nRanks;
		size_t nNodes = ctx->nNodes;

		/* Only tune for 0x7 topology (nRanks == nNodes) */
		if (nRanks != nNodes) {
			return ncclSuccess;
		}

		// AllReduce 0x7 Tree LL128 chunk size tuning (aws-ofi-nccl, P5en)
		size_t nstepsLL128 = 1 + log2i(nNodes);

		size_t tunedChunkSize = 288000; // maxChunkSize

		// Step 1: 288000 → 144000 (threshold: 2 × nsteps²)
		if (nBytes / tunedChunkSize < 2 * nstepsLL128 * nstepsLL128)
			tunedChunkSize /= 2;

		// Step 2: 144000 → 72000 (threshold: 1.5 × nsteps)
		if (nBytes / tunedChunkSize < 1.5 * nstepsLL128)
			tunedChunkSize /= 2;

		// Step 3: 72000 → 36000 → 18000 (threshold: nsteps + 1)
		while (nBytes / tunedChunkSize < nstepsLL128 + 1 && tunedChunkSize > 18000)
			tunedChunkSize /= 2;

		NCCL_OFI_INFO(NCCL_TUNING,
			"getChunkSize: AllReduce Tree/LL128 nBytes=%zu nNodes=%zu chunkSize=%zu -> %zu",
			nBytes, nNodes, *chunkSize, tunedChunkSize);

		*chunkSize = tunedChunkSize;
	}

	if (collType == ncclFuncAllGather && algo == NCCL_ALGO_PAT && proto == NCCL_PROTO_SIMPLE) {
		int nRanks = (int)ctx->nRanks;
		int nNodes = (int)ctx->nNodes;

		/* Only tune for 0x7 topology (nRanks == nNodes) */
		if (nRanks != nNodes) {
			return ncclSuccess;
		}

		// AllGather 0x7 PAT Simple chunk size tuning (aws-ofi-nccl, P5en)
		// Optimized for 16N/32N accuracy: 16N uses same chunk sizes as 32N from 512K–64M
		int satChunkSize = nNodes >= 16 ? 524288 : 1048576;
		size_t T1 = nNodes >= 16 ? 32 : std::min(nNodes, 8) + 1;
		size_t T2 = std::min(nNodes, 16);
		size_t T3 = std::min(nNodes, 8);

		int tunedChunkSize = satChunkSize;

		// Step 1: cap — halve from saturation ceiling at the transition zone
		while (tunedChunkSize * T1 > nBytes && tunedChunkSize > satChunkSize / 2)
			tunedChunkSize /= 2;

		// Step 2: mid — halve through the mid-range chunk sizes (down to 64K)
		while (tunedChunkSize * T2 > nBytes && tunedChunkSize > 65536)
			tunedChunkSize /= 2;

		// Step 3: low — halve through the smallest chunk sizes (down to 32K)
		while (tunedChunkSize * T3 > nBytes && tunedChunkSize > 32768)
			tunedChunkSize /= 2;

		NCCL_OFI_INFO(NCCL_TUNING,
			"getChunkSize: AllGather PAT/Simple nBytes=%zu nNodes=%d chunkSize=%zu -> %d",
			nBytes, nNodes, *chunkSize, tunedChunkSize);

		*chunkSize = (size_t)tunedChunkSize;
	}

	return ncclSuccess;
}

extern "C" const ncclTuner_v6_t ncclTunerPlugin_v6 = {.name = "nccl_ofi_tuner",
					   .init = nccl_ofi_tuner_init_v6,
					   .getCollInfo = nccl_ofi_tuner_get_coll_info_v6,
					   .finalize = nccl_ofi_tuner_finalize,
					   .getChunkSize = nccl_ofi_tuner_get_chunk_size};

/* **** V2 **** */
static ncclResult_t nccl_ofi_tuner_get_coll_info_v2(
	void *context, ncclFunc_t collType, size_t nBytes, int collNetSupport, int nvlsSupport, int numPipeOps, int *algorithm, int *protocol, int *nChannels)
{
	ncclResult_t ret;

	nccl_ofi_tuner_context_t *ctx = (nccl_ofi_tuner_context_t *)context;
	if (ctx == NULL || ctx->get_coll_info_internal_v2 == NULL) {
		/* Fall back to NCCL's tuner */
		return ncclSuccess;
	}

	ret = ctx->get_coll_info_internal_v2(ctx,
					     collType,
					     nBytes,
					     collNetSupport,
					     nvlsSupport,
					     numPipeOps,
					     algorithm,
					     protocol,
					     nChannels);

	return ret;
}

extern "C" const ncclTuner_v2_t ncclTunerPlugin_v2 = {.name = "nccl_ofi_tuner",
					   .init = nccl_ofi_tuner_init_v2,
					   .getCollInfo = nccl_ofi_tuner_get_coll_info_v2,
					   .destroy = nccl_ofi_tuner_destroy};

/* **** V1 ****
 * The tuner v1 API is missing a mechanism to pass around context after
 * initialization. For now, init a plugin-global context once.
 */
static nccl_ofi_tuner_context_t *nccl_ofi_tuner_ctx_internal;

static ncclResult_t nccl_ofi_tuner_destroy_v1(void)
{
	void *context = NULL;

	nccl_net_ofi_mutex_lock(&nccl_ofi_tuner_ctx_lock);
	if (nccl_ofi_tuner_ctx_internal != NULL) {
		/* Prevent other threads from freeing a dangling global ctx */
		context = (void *)nccl_ofi_tuner_ctx_internal;
		nccl_ofi_tuner_ctx_internal = NULL;
	}
	nccl_net_ofi_mutex_unlock(&nccl_ofi_tuner_ctx_lock);

	return nccl_ofi_tuner_destroy(context);
}

static ncclResult_t nccl_ofi_tuner_init_v1(size_t nRanks, size_t nNodes, ncclDebugLogger_t logFunction)
{
	if (nccl_ofi_tuner_ctx_internal != NULL) {
		/* Repeated init call, the tuner is already initialized.
		 * Destroy it, as it may have been initialized with different
		 * parameters.
		 */
		if (nccl_ofi_tuner_destroy_v1() != ncclSuccess) {
			NCCL_OFI_WARN(
				"Failed to destroy an existing tuner context.");
		}
	}

	/*
	 * NCCL parses these variables and applies user filters inside its
	 * current tuner logic. The tuner_v1 does not support setting these
	 * variables and so the internal tuner will be used instead.
	 */
	if (getenv("NCCL_ALGO") || getenv("NCCL_PROTO")) {
		NCCL_OFI_INFO(NCCL_INIT | NCCL_TUNING, "The tuner plugin can not be loaded when "
				"explicitly choosing an algorithm or protocol "
				"with NCCL_ALGO/NCCL_PROTO. "
				"Defaulting to internal tuner.");
		nccl_ofi_tuner_destroy_v1();
		return ncclSuccess;
	}
	return nccl_ofi_tuner_init(nRanks, nNodes, logFunction, (void **)&nccl_ofi_tuner_ctx_internal);
}

static ncclResult_t nccl_ofi_tuner_get_coll_info_v1(
	ncclFunc_t collType, size_t nBytes, int collNetSupport, int nvlsSupport, int numPipeOps, int *algorithm, int *protocol, int *nChannels)
{
	return nccl_ofi_tuner_get_coll_info_v2(nccl_ofi_tuner_ctx_internal,
					    collType,
					    nBytes,
					    collNetSupport,
					    nvlsSupport,
					    numPipeOps,
					    algorithm,
					    protocol,
					    nChannels);
}

extern "C" const ncclTuner_v1_t ncclTunerPlugin_v1 = {.name = "nccl_ofi_tuner",
					   .init = nccl_ofi_tuner_init_v1,
					   .getCollInfo = nccl_ofi_tuner_get_coll_info_v1,
					   .destroy = nccl_ofi_tuner_destroy_v1};
