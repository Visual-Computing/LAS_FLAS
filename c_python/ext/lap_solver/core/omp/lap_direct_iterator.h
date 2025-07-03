#pragma once

#include "lap_worksharing.h"

namespace lap
{
	namespace omp
	{
		template <class TC, class CF>
		class DirectIterator
		{
		public:
			CF &costfunc;
			Worksharing &ws;

		public:
			DirectIterator(CF &costfunc, Worksharing &ws) : costfunc(costfunc), ws(ws) {}
			~DirectIterator() {}

			void getHitMiss(long long &hit, long long &miss) { hit = miss = 0; }

			__forceinline const TC *getRow(int t, int i) { return costfunc.getRow(t, i); }
		};
	}
}
