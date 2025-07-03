#pragma once

#if defined(__GNUC__)
#define __forceinline \
        __inline__ __attribute__((always_inline))
#endif

#ifdef LAP_QUIET
#define lapAssert(A)
#else
// note: this will automatically be disabled if NDEBUG is set
#include <assert.h>
#define lapAssert(A) assert(A)
#endif

#ifndef lapInfo
#define lapInfo std::cout
#endif

#ifndef lapDebug
#define lapDebug std::cout
#endif

#ifndef lapAlloc
#define lapAlloc lap::alloc
#endif

#ifndef lapFree
#define lapFree lap::free
#endif

#ifdef LAP_CUDA
#ifndef lapAllocPinned
#define lapAllocPinned lap::cuda::allocPinned
#endif

#ifndef lapFreePinned
#define lapFreePinned lap::cuda::freePinned
#endif

#ifndef lapAllocDevice
#define lapAllocDevice lap::cuda::allocDevice
#endif

#ifndef lapFreeDevice
#define lapFreeDevice lap::cuda::freeDevice
#endif
#endif

#include <tuple>

namespace lap
{
	// Functions used for solving the lap, calculating the costs of a certain assignment and guessing the initial epsilon value.
	template <class SC, class CF, class I> void solve(int dim, CF &costfunc, I &iterator, int *rowsol, bool use_epsilon);
	template <class SC, class CF, class I> void solve(int dim, int dim2, CF &costfunc, I &iterator, int *rowsol, bool use_epsilon);
	template <class SC, class CF> SC cost(int dim, CF &costfunc, int *rowsol);
	template <class SC, class CF> SC cost(int dim, int dim2, CF &costfunc, int *rowsol);

	// Cost functions, including tabulated costs
	template <class TC, typename GETCOST> class SimpleCostFunction;
	template <class TC, typename GETCOSTROW> class RowCostFunction;
	template <class TC> class TableCost;

	// Iterator classes used for accessing the cost functions
	template <class TC, class CF> class DirectIterator;
	template <class TC, class CF, class CACHE> class CachingIterator;

	// Caching Schemes to be used for caching iterator
	class CacheSLRU;
	class CacheLFU;

	// Memory management
	template <typename T> void alloc(T * &ptr, unsigned long long width, const char *file, const int line);
	template <typename T> void free(T *&ptr);

#ifdef LAP_OPENMP
	namespace omp
	{
		// Functions used for solving the lap, calculating the costs of a certain assignment and guessing the initial epsilon value.
		template <class SC, class CF, class I> void solve(int dim, CF &costfunc, I &iterator, int *rowsol, bool use_epsilon);
		template <class SC, class CF, class I> void solve(int dim, int dim2, CF &costfunc, I &iterator, int *rowsol, bool use_epsilon);
		template <class SC, class CF> SC cost(int dim, CF &costfunc, int *rowsol);
		template <class SC, class CF> SC cost(int dim, int dim2, CF &costfunc, int *rowsol);

		// Cost functions, including tabulated costs
		template <class TC, typename GETCOST> class SimpleCostFunction;
		template <class TC, typename GETENABLED, typename GETCOSTROW> class RowCostFunction;
		template <class TC> class TableCost;

		// Iterator classes used for accessing the cost functions
		template <class TC, class CF> class DirectIterator;
		template <class TC, class CF, class CACHE> class CachingIterator;

	}
#endif
}

#include "core/lap_cost.h"
#include "core/lap_cache.h"
#include "core/lap_direct_iterator.h"
#include "core/lap_caching_iterator.h"
#include "core/lap_solver.h"

#ifdef LAP_OPENMP
#include <omp.h>
#include "core/omp/lap_cost.h"
#include "core/omp/lap_direct_iterator.h"
#include "core/omp/lap_caching_iterator.h"
#include "core/omp/lap_solver.h"
#endif
