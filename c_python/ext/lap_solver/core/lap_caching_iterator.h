#pragma once

#include "lap_direct_iterator.h"

namespace lap
{
	template <class TC, class CF, class CACHE>
	class CachingIterator
	{
	protected:
		TC* rows;
		CACHE cache;
	public:
		int dim2;
		CF &costfunc;
	public:
		CachingIterator(int dim, int dim2, int entries, CF &costfunc)
			: dim2(dim2), costfunc(costfunc)
		{
			cache.setSize(entries, dim);
			lapAlloc(rows, (long long)entries * (long long)dim2, __FILE__, __LINE__);
		}

		~CachingIterator()
		{
			lapFree(rows);
		}

		__forceinline void getHitMiss(long long &hit, long long &miss) { cache.getHitMiss(hit, miss); }

		__forceinline const TC *getRow(int i)
		{
			int idx;
			bool found = cache.find(idx, i);
			if (!found)
			{
				costfunc.getCostRow(rows + (long long)dim2 * (long long)idx, i, 0, dim2);
			}
			return rows + (long long)dim2 * (long long)idx;
		}
	};
}
