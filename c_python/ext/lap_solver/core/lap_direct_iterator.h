#pragma once

#include <algorithm>

namespace lap
{
	template <class TC, class CF>
	class DirectIterator
	{
	public:
		CF &costfunc;
	public:
		DirectIterator(CF &costfunc) : costfunc(costfunc) {}
		~DirectIterator() {}

		void getHitMiss(long long &hit, long long &miss) { hit = miss = 0; }

		__forceinline const TC *getRow(int i) { return costfunc.getRow(i); }
	};
}
