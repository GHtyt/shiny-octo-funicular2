#ifndef SOS_BASE_H_
#define SOS_BASE_H_

#define __CUDA__

#if defined (__CUDA__) || defined(__NVCC__)
#define XGBOOST_DEVICE __host__ __device__
#define XGBOOST_HOST_DEV_INLINE XGBOOST_DEVICE __forceinline__
#define XGBOOST_DEV_INLINE __device__ __forceinline__
#else
#define XGBOOST_DEVICE
#define XGBOOST_HOST_DEV_INLINE XGBOOST_DEVICE 
#define XGBOOST_DEV_INLINE 
#endif 


#include <iostream>

#define XGBOOST_EXPECT(cond, ret)  __assume((cond), (ret))
//#define XGBOOST_EXPECT(cond, ret) (cond)

using bst_uint = uint32_t;  // NOLINT
/*! \brief integer type. */
using bst_int = int32_t;    // NOLINT
/*! \brief unsigned long integers */
using bst_ulong = uint64_t;  // NOLINT
/*! \brief float type, used for storing statistics */
using bst_float = float;  // NOLINT
/*! \brief Categorical value type. */
using bst_cat_t = int32_t;  // NOLINT
/*! \brief Type for data column (feature) index. */
using bst_feature_t = uint32_t;  // NOLINT
/*! \brief Type for data row index.
using bst_row_t = std::size_t;   // NOLINT
/*! \brief Type for tree node index. */
using bst_node_t = int32_t;      // NOLINT
/*! \brief Type for ranking group index. */
using bst_group_t = uint32_t;    // NOLINT


#endif