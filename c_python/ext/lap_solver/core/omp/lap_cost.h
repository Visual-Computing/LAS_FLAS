#pragma once

#include "lap_worksharing.h"
#include "../lap_cost.h"

namespace lap
{
	namespace omp
	{
		// Wrapper around simple cost function, scheduling granularity is assumed to be 8 for load balancing
		template <class TC, typename GETCOST>
		class SimpleCostFunction : public lap::SimpleCostFunction<TC, GETCOST>
		{
		protected:
			bool sequential;
		public:
			SimpleCostFunction(GETCOST &getcost, bool sequential = false) : lap::SimpleCostFunction<TC, GETCOST>(getcost), sequential(sequential) {}
			~SimpleCostFunction() {}
		public:
			__forceinline int getMultiple() const { return 8; }
			__forceinline bool isSequential() const { return sequential; }
		};

		// Costs stored in a table. Used for conveniency only
		// This can be constructed using a CostFunction from above or by specifying an array that holds the data (does not copy the data in this case).
		template <class TC>
		class TableCost
		{
		protected:
			int x_size;
			int y_size;
			TC **cc;
			int *stride;
			bool free_in_destructor;
			Worksharing &ws;
		protected:
			void referenceTable(TC *tab)
			{
				free_in_destructor = false;
				lapAlloc(cc, omp_get_max_threads(), __FILE__, __LINE__);
				lapAlloc(stride, omp_get_max_threads(), __FILE__, __LINE__);
				for (int t = 0; t < omp_get_max_threads(); t++)
				{
					stride[t] = y_size;
					cc[t] = &(tab[ws.part[t].first]);
				}
			}
			
			template <class DirectCost>
			void initTable(DirectCost &cost)
			{
				free_in_destructor = true;
				lapAlloc(cc, omp_get_max_threads(), __FILE__, __LINE__);
				lapAlloc(stride, omp_get_max_threads(), __FILE__, __LINE__);
				if (cost.isSequential())
				{
					// cost table needs to be initialized sequentially
#pragma omp parallel
					{
						const int t = omp_get_thread_num();
						stride[t] = ws.part[t].second - ws.part[t].first;
						lapAlloc(cc[t], (long long)(stride[t]) * (long long)x_size, __FILE__, __LINE__);
						for (int x = 0; x < x_size; x++)
						{
							for (int tt = 0; tt < omp_get_max_threads(); tt++)
							{
#pragma omp barrier
								if (tt == t) cost.getCostRow(cc[t] + (long long)x * (long long)stride[t], x, ws.part[t].first, ws.part[t].second);
							}
						}
					}
				}
				else
				{
					// create and initialize in parallel
#pragma omp parallel
					{
						const int t = omp_get_thread_num();
						stride[t] = ws.part[t].second - ws.part[t].first;
						lapAlloc(cc[t], (long long)(stride[t]) * (long long)x_size, __FILE__, __LINE__);
						// first touch
						cc[t][0] = TC(0);
						for (int x = 0; x < x_size; x++)
						{
							cost.getCostRow(cc[t] + (long long)x * (long long)stride[t], x, ws.part[t].first, ws.part[t].second);
						}
					}
				}
			}

			void createTable()
			{
				free_in_destructor = true;
				lapAlloc(cc, omp_get_max_threads(), __FILE__, __LINE__);
				lapAlloc(stride, omp_get_max_threads(), __FILE__, __LINE__);
#pragma omp parallel
				{
					const int t = omp_get_thread_num();
					stride[t] = ws.part[t].second - ws.part[t].first;
					lapAlloc(cc[t], (long long)(stride[t]) * (long long)x_size, __FILE__, __LINE__);
					// first touch
					cc[t][0] = TC(0);
				}
			}
		public:
			template <class DirectCost> TableCost(int x_size, int y_size, DirectCost &cost, Worksharing &ws) :
				x_size(x_size), y_size(y_size), ws(ws) { initTable(cost); }
			template <class DirectCost> TableCost(int size, DirectCost &cost, Worksharing &ws) :
				x_size(size), y_size(size), ws(ws) { initTable(cost); }
			TableCost(int x_size, int y_size, Worksharing &ws) : x_size(x_size), y_size(y_size), ws(ws) { createTable(); }
			TableCost(int size, Worksharing &ws) : x_size(size), y_size(size), ws(ws) { createTable(); }
			TableCost(int x_size, int y_size, TC* tab, Worksharing &ws) : x_size(x_size), y_size(y_size), ws(ws) { referenceTable(tab); }
			TableCost(int size, TC* tab, Worksharing &ws) : x_size(size), y_size(size), ws(ws) { referenceTable(tab); }
			~TableCost()
			{
				if (free_in_destructor)
				{
#pragma omp parallel
					lapFree(cc[omp_get_thread_num()]);
				}
				lapFree(cc);
				lapFree(stride);
			}
			public:
			__forceinline const TC *getRow(int t, int x) const { return cc[t] + (long long)x * (long long)stride[t]; }
			__forceinline const TC getCost(int x, int y) const
			{
				int t = 0;
				while (y >= ws.part[t].second) t++;
				long long off_y = y - (long long)ws.part[t].first;
				long long off_x = x;
				off_x *= stride[t];
				return cc[t][off_x + off_y];
			}
			__forceinline void setRow(int x, TC *v)
			{
				for (int t = 0; t < omp_get_max_threads(); t++)
				{
					long long off_x = x;
					off_x *= stride[t];
					memcpy(&(cc[t][off_x]), &(v[ws.part[t].first]), (ws.part[t].second - ws.part[t].first) * sizeof(TC));
				}
			}
		};
	}
}
