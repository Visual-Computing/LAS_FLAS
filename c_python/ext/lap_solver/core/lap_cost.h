#pragma once

#include <string.h>

namespace lap
{
	// Wrapper around simple cost function
	template <class TC, typename GETCOST>
	class SimpleCostFunction
	{
	protected:
		GETCOST getcost;
	public:
		SimpleCostFunction(GETCOST &getcost) : getcost(getcost) {}
		~SimpleCostFunction() {}
	public:
		__forceinline const TC getCost(int x, int y) const { return getcost(x, y); }
		__forceinline void getCostRow(TC *row, int x, int start, int end) const { for (int y = start; y < end; y++) row[y - start] = getCost(x, y); }
	};

	// Wrapper around per-row cost funtion, e.g. CUDA, OpenCL or OpenMPI
	template <class TC, typename GETCOSTROW>
	class RowCostFunction
	{
	protected:
		GETCOSTROW getcostrow;
	public:
		RowCostFunction(GETCOSTROW &getcostrow) : getcostrow(getcostrow) {}
		~RowCostFunction() {}
	public:
		__forceinline const TC getCost(int x, int y) const {
			TC r;
			getcostrow(&r, x, y, y + 1);
			return r;
		}
		__forceinline void getCostRow(TC *row, int x, int start, int end) const { getcostrow(row, x, start, end); }
	};

	// Costs stored in a table. Used for conveniency only
	// This can be constructed using a CostFunction from above or by specifying an array that holds the data (does not copy the data in this case).
	template <class TC>
	class TableCost
	{
	protected:
		int x_size;
		int y_size;
		TC *c;
		bool free_in_destructor;
	protected:
		template <class DirectCost>
		void initTable(DirectCost &cost)
		{
			lapAlloc(c, (long long)x_size * (long long)y_size, __FILE__, __LINE__);
			free_in_destructor = true;
			for (int x = 0; x < x_size; x++)
			{
				cost.getCostRow(&(c[(long long)x * (long long)y_size]), x, 0, y_size);
			}
		}
		void createTable()
		{
			lapAlloc(c, (long long)x_size * (long long)y_size, __FILE__, __LINE__);
			free_in_destructor = true;
		}
	public:
		template <class DirectCost> TableCost(int x_size, int y_size, DirectCost &cost) : x_size(x_size), y_size(y_size) { initTable(cost); }
		template <class DirectCost> TableCost(int size, DirectCost &cost) : x_size(size), y_size(size) { initTable(cost); }
		TableCost(int x_size, int y_size) : x_size(x_size), y_size(y_size) { createTable(); }
		TableCost(int size) : x_size(size), y_size(size) { createTable(); }
		TableCost(int x_size, int y_size, TC* tab) : x_size(x_size), y_size(y_size), c(tab) { free_in_destructor = false; }
		TableCost(int size, TC* tab) : x_size(size), y_size(size), c(tab) { free_in_destructor = false; }
		~TableCost() { if (free_in_destructor) lapFree(c); }
	public:
		__forceinline const TC *getRow(int x) const { return &(c[(long long)x * (long long)y_size]); }
		__forceinline const TC getCost(int x, int y) const { return getRow(x)[y]; }
		__forceinline void setRow(int x, TC *v) { memcpy(&(c[(long long)x * (long long)y_size]), v, y_size * sizeof(TC)); }
	};
}
