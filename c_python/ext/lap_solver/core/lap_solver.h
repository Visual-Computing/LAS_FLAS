#pragma once
#pragma once

#include <chrono>
#include <sstream>
#include <iostream>
#include <cstring>
#ifndef LAP_QUIET
#include <deque>
#include <mutex>
#endif
#include <math.h>

namespace lap
{
#ifndef LAP_QUIET
	class AllocationLogger
	{
		std::vector<std::deque<void *>> allocated;
		std::vector<std::deque<unsigned long long>> size;
		std::vector<std::deque<char *>> alloc_file;
		std::vector<std::deque<int>> alloc_line;
		std::vector<unsigned long long> peak;
		std::vector<unsigned long long> current;
		std::vector<std::string> name;
		std::mutex lock;
	private:
		std::string commify(unsigned long long n)
		{
			std::string s;
			int cnt = 0;
			do
			{
				s.insert(0, 1, char('0' + n % 10));
				n /= 10;
				if (++cnt == 3 && n)
				{
					s.insert(0, 1, ',');
					cnt = 0;
				}
			} while (n);
			return s;
		}

	public:
		AllocationLogger()
		{
#ifdef LAP_CUDA
			allocated.resize(3);
			size.resize(3);
			alloc_file.resize(3);
			alloc_line.resize(3);
			peak.resize(3);
			current.resize(3);
			name.resize(3);
			peak[0] = 0ull; peak[1] = 0ull; peak[2] = 0ull;
			current[0] = 0ull; current[1] = 0ull; current[2] = 0ull;
			name[0] = std::string("system memory");
			name[1] = std::string("pinned memory");
			name[2] = std::string("device memory");
#else
			allocated.resize(1);
			size.resize(1);
			alloc_file.resize(1);
			alloc_line.resize(1);
			peak.resize(1);
			current.resize(1);
			name.resize(1);
			peak[0] = 0ull;
			current[0] = 0ull;
			name[0] = std::string("memory");
#endif
		}

		~AllocationLogger() {}
		void destroy()
		{
			for (size_t i = 0; i < peak.size(); i++)
			{
				if (!name[i].empty())
				{
					lapInfo << "Peak " << name[i] << " usage:" << commify(peak[i]) << " bytes" << std::endl;
					if (allocated[i].empty()) continue;
					lapInfo << (char)toupper(name[i][0]) << name[i].substr(1) << " leak list:" << std::endl;
					while (!allocated[i].empty())
					{
						lapInfo << "  leaked " << commify(size[i].front()) << " bytes at " << std::hex << allocated[i].front() << std::dec << ": " << alloc_file[i].front() << ":" << alloc_line[i].front() << std::endl;
						size[i].pop_front();
						allocated[i].pop_front();
						alloc_file[i].pop_front();
						alloc_line[i].pop_front();
					}
				}
			}
		}

		template <class T>
		void free(int idx, T a)
		{
			std::lock_guard<std::mutex> guard(lock);
#ifdef LAP_DEBUG
#ifndef LAP_NO_MEM_DEBUG
			lapDebug << "Freeing memory at " << std::hex << (size_t)a << std::dec << std::endl;
#endif
#endif
			for (unsigned long long i = 0; i < allocated[idx].size(); i++)
			{
				if ((void *)a == allocated[idx][i])
				{
					current[idx] -= size[idx][i];
					allocated[idx][i] = allocated[idx].back();
					allocated[idx].pop_back();
					size[idx][i] = size[idx].back();
					size[idx].pop_back();
					alloc_line[idx][i] = alloc_line[idx].back();
					alloc_line[idx].pop_back();
					alloc_file[idx][i] = alloc_file[idx].back();
					alloc_file[idx].pop_back();
					return;
				}
			}
		}

		template <class T>
		void alloc(int idx, T *a, unsigned long long s, const char *file, const int line)
		{
			std::lock_guard<std::mutex> guard(lock);
#ifdef LAP_DEBUG
#ifndef LAP_NO_MEM_DEBUG
			lapDebug << "Allocating " << s * sizeof(T) << " bytes at " << std::hex << (size_t)a << std::dec << " \"" << file << ":" << line << std::endl;
#endif
#endif
			current[idx] += s * sizeof(T);
			peak[idx] = std::max(peak[idx], current[idx]);
			allocated[idx].push_back((void *)a);
			size[idx].push_back(s * sizeof(T));
			alloc_file[idx].push_back((char *)file);
			alloc_line[idx].push_back(line);
		}
	};

	static AllocationLogger allocationLogger;
#endif

	template <typename T>
	void alloc(T * &ptr, unsigned long long width, const char *file, const int line)
	{
		ptr = (T*)malloc(sizeof(T) * (size_t) width); // this one is allowed
#ifndef LAP_QUIET
		allocationLogger.alloc(0, ptr, width, file, line);
#endif
	}

	template <typename T>
	void free(T *&ptr)
	{
		if (ptr == (T *)NULL) return;
#ifndef LAP_QUIET
		allocationLogger.free(0, ptr);
#endif
		::free(ptr); // this one is allowed
		ptr = (T *)NULL;
	}

	std::string getTimeString(long long ms)
	{
		char time[256];
		long long sec = ms / 1000;
		ms -= sec * 1000;
		long long min = sec / 60;
		sec -= min * 60;
		long long hrs = min / 60;
		min -= hrs * 60;
#if defined (_MSC_VER)
		sprintf_s(time, "%3d:%02d:%02d.%03d", (int)hrs, (int)min, (int)sec, (int)ms);
#else
		sprintf(time, "%3d:%02d:%02d.%03d", (int)hrs, (int)min, (int)sec, (int)ms);
#endif

		return std::string(time);
	}

	std::string getSecondString(long long ms)
	{
		char time[256];
		long long sec = ms / 1000;
		ms -= sec * 1000;
#if defined (_MSC_VER)
		sprintf_s(time, "%d.%03d", (int)sec, (int)ms);
#else
		sprintf(time, "%d.%03d", (int)sec, (int)ms);
#endif

		return std::string(time);
	}

	template <class TP, class OS>
	void displayTime(TP &start_time, const char *msg, OS &lapStream)
	{
		auto end_time = std::chrono::high_resolution_clock::now();
		long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
		lapStream << getTimeString(ms) << ": " << msg << " (" << getSecondString(ms) << "s)" << std::endl;
	}

	template <class TP>
	int displayProgress(TP &start_time, int &elapsed, int completed, int target_size, const char *msg = 0, int iteration = -1, bool display = false)
	{
		if (completed == target_size) display = true;

#ifndef LAP_DEBUG
		if (!display) return 0;
#endif

		auto end_time = std::chrono::high_resolution_clock::now();
		long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

#ifdef LAP_DEBUG
		if ((!display) && (elapsed * 10000 < ms))
		{
			elapsed = (int)((ms + 10000ll) / 10000ll);
			lapDebug << getTimeString(ms) << ": solving " << completed << "/" << target_size;
			if (iteration >= 0) lapDebug << " iteration = " << iteration;
			if (msg != 0) lapDebug << msg;
			lapDebug << std::endl;
			return 2;
		}

		if (display)
#endif
		{
			elapsed = (int)((ms + 10000ll) / 10000ll);
			lapInfo << getTimeString(ms) << ": solving " << completed << "/" << target_size;
			if (iteration >= 0) lapInfo << " iteration = " << iteration;
			if (msg != 0) lapInfo << msg;
			lapInfo << std::endl;
			return 1;
		}
#ifdef LAP_DEBUG
		return 0;
#endif
	}

	template <class SC, typename COST>
	void getMinMaxBest(int i, SC &min_cost_l, SC &max_cost_l, SC &picked_cost_l, int &j_min, COST &cost, int *taken, int count)
	{
		max_cost_l = min_cost_l = cost(0);
		if (taken[0] == 0)
		{
			j_min = 0;
			picked_cost_l = min_cost_l;
		}
		else
		{
			j_min = std::numeric_limits<int>::max();
			picked_cost_l = std::numeric_limits<SC>::max();
		}
		for (int j = 1; j < count; j++)
		{
			SC cost_l = cost(j);
			min_cost_l = std::min(min_cost_l, cost_l);
			if (i == j) max_cost_l = cost_l;
			if ((cost_l < picked_cost_l) && (taken[j] == 0))
			{
				j_min = j;
				picked_cost_l = cost_l;
			}
		}
	}

	template <class SC, typename COST>
	void getMinSecondBest(SC &min_cost_l, SC &second_cost_l, SC &picked_cost_l, int &j_min, COST &cost, int *taken, int count)
	{
		min_cost_l = std::min(cost(0), cost(1));
		second_cost_l = std::max(cost(0), cost(1));
		if ((taken[0] == 0) && (taken[1] == 0))
		{
			picked_cost_l = min_cost_l;
			if (cost(0) == min_cost_l)
			{
				j_min = 0;
			}
			else
			{
				j_min = 1;
			}
		}
		else if (taken[0] == 0)
		{
			j_min = 0;
			picked_cost_l = cost(0);
		}
		else if (taken[1] == 0)
		{
			j_min = 1;
			picked_cost_l = cost(1);
		}
		else
		{
			j_min = std::numeric_limits<int>::max();
			picked_cost_l = std::numeric_limits<SC>::max();
		}
		for (int j = 2; j < count; j++)
		{
			SC cost_l = cost(j);
			if (cost_l < min_cost_l)
			{
				second_cost_l = min_cost_l;
				min_cost_l = cost_l;
			}
			else second_cost_l = std::min(second_cost_l, cost_l);
			if ((cost_l < picked_cost_l) && (taken[j] == 0))
			{
				j_min = j;
				picked_cost_l = cost_l;
			}
		}
	}

	template <class SC, typename COST>
	void updateEstimatedV(SC* v, SC *min_v, COST &cost, bool first, bool second, SC min_cost_l, SC max_cost_l, int count)
	{
		if (first)
		{
			for (int j = 0; j < count; j++)
			{
				SC tmp = cost(j) - min_cost_l;
				min_v[j] = tmp;
			}
		}
		else if (second)
		{
			for (int j = 0; j < count; j++)
			{
				SC tmp = cost(j) - min_cost_l;
				if (tmp < min_v[j])
				{
					v[j] = min_v[j];
					min_v[j] = tmp;
				}
				else v[j] = tmp;
			}
		}
		else
		{
			for (int j = 0; j < count; j++)
			{
				SC tmp = cost(j) - min_cost_l;
				if (tmp < min_v[j])
				{
					v[j] = min_v[j];
					min_v[j] = tmp;
				}
				else v[j] = std::min(v[j], tmp);
			}
		}
	}

	template <class SC>
	void normalizeV(SC *v, int count, int *colsol)
	{
		SC max_v = std::numeric_limits<SC>::lowest();
		for (int j = 0; j < count; j++) if (colsol[j] >= 0) max_v = std::max(max_v, v[j]);
		for (int j = 0; j < count; j++) v[j] = std::min(SC(0), v[j] - max_v);
	}

	template <class SC>
	void normalizeV(SC *v, int count)
	{
		SC max_v = v[0];
		for (int j = 1; j < count; j++) max_v = std::max(max_v, v[j]);
		for (int j = 0; j < count; j++) v[j] = v[j] - max_v;
	}

	template <class SC, typename COST>
	void getMinimalCost(int &j_min, SC &min_cost, SC &min_cost_real, COST &cost, SC *mod_v, int count)
	{
		j_min = std::numeric_limits<int>::max();
		min_cost = std::numeric_limits<SC>::max();
		min_cost_real = std::numeric_limits<SC>::max();
		for (int j = 0; j < count; j++)
		{
			SC cost_l = cost(j);
			if (mod_v[j] < SC(0))
			{
				if (cost_l < min_cost)
				{
					min_cost = cost_l;
					j_min = j;
				}
			}
			min_cost_real = std::min(min_cost_real, cost_l);
		}
	}

	template<typename SC>
	void getUpperLower(SC& upper, SC& lower, double greedy_gap, double initial_gap, int dim, int dim2)
	{
		greedy_gap = std::min(greedy_gap, initial_gap / 4.0);
		if (greedy_gap < 1.0e-6 * initial_gap) upper = SC(0);
		else upper = (SC)((double)dim * greedy_gap * sqrt(greedy_gap / initial_gap) / ((double)dim2 * (double)dim2));
		lower = (SC)(initial_gap / (16.0 * (double)dim2 * (double)dim2));
		if (upper < lower) upper = lower = SC(0);
	}

	template <class SC, class I>
	std::pair<SC, SC> estimateEpsilon(int dim, int dim2, I& iterator, SC *v, int *perm)
	{
#ifdef LAP_DEBUG
		auto start_time = std::chrono::high_resolution_clock::now();
#endif
		SC *mod_v;
		int *picked;
		SC *v2;

		lapAlloc(mod_v, dim2, __FILE__, __LINE__);
		lapAlloc(v2, dim2, __FILE__, __LINE__);
		lapAlloc(picked, dim2, __FILE__, __LINE__);

		double lower_bound = 0.0;
		double greedy_bound = 0.0;
		double upper_bound = 0.0;

		memset(picked, 0, sizeof(int) * dim2);

		for (int i = 0; i < dim2; i++)
		{
			SC min_cost_l, max_cost_l, picked_cost_l;
			int j_min;
			if (i < dim)
			{
				const auto *tt = iterator.getRow(i);
				auto cost = [&tt](int j) -> SC { return (SC)tt[j]; };
				getMinMaxBest(i, min_cost_l, max_cost_l, picked_cost_l, j_min, cost, picked, dim2);
				picked[j_min] = 1;
				updateEstimatedV(v, mod_v, cost, (i == 0), (i == 1), min_cost_l, max_cost_l, dim2);
				lower_bound += min_cost_l;
				upper_bound += max_cost_l;
				greedy_bound += picked_cost_l;
			}
			else
			{
				auto cost = [](int j) -> SC { return SC(0); };
				getMinMaxBest(i, min_cost_l, max_cost_l, picked_cost_l, j_min, cost, picked, dim2);
				picked[j_min] = 1;
				updateEstimatedV(v, mod_v, cost, (i == 0), (i == 1), min_cost_l, max_cost_l, dim2);
				lower_bound += min_cost_l;
				greedy_bound += picked_cost_l;
			}
		}
		// make sure all j are < 0
		normalizeV(v, dim2);

		greedy_bound = std::min(greedy_bound, upper_bound);

		double initial_gap = upper_bound - lower_bound;
		double greedy_gap = greedy_bound - lower_bound;
		double initial_greedy_gap = greedy_gap;

#ifdef LAP_DEBUG
		{
			std::stringstream ss;
			ss << "upper_bound = " << upper_bound << " lower_bound = " << lower_bound << " initial_gap = " << initial_gap;
			lap::displayTime(start_time, ss.str().c_str(), lapDebug);
		}
		{
			std::stringstream ss;
			ss << "upper_bound = " << greedy_bound << " lower_bound = " << lower_bound << " greedy_gap = " << greedy_gap << " ratio = " << greedy_gap / initial_gap;
			lap::displayTime(start_time, ss.str().c_str(), lapDebug);
		}
#endif

		memset(picked, 0, sizeof(int) * dim2);

		lower_bound = 0.0;
		upper_bound = 0.0;

		// reverse order
		for (int i = dim2 - 1; i >= 0; --i)
		{
			SC min_cost_l, second_cost_l, picked_cost_l;
			int j_min;
			if (i < dim)
			{
				const auto *tt = iterator.getRow(i);
				auto cost = [&tt, &v](int j) -> SC { return (SC)tt[j] - v[j]; };
				getMinSecondBest(min_cost_l, second_cost_l, picked_cost_l, j_min, cost, picked, dim2);
			}
			else
			{
				auto cost = [&v](int j) -> SC { return -v[j]; };
				getMinSecondBest(min_cost_l, second_cost_l, picked_cost_l, j_min, cost, picked, dim2);
			}
			perm[i] = i;
			picked[j_min] = 1;
			mod_v[i] = second_cost_l - min_cost_l;
			// need to use the same v values in total
			lower_bound += min_cost_l + v[j_min];
			upper_bound += picked_cost_l + v[j_min];
		}

		upper_bound = greedy_bound = std::min(upper_bound, greedy_bound);

		greedy_gap = upper_bound - lower_bound;

#ifdef LAP_DEBUG
		{
			std::stringstream ss;
			ss << "upper_bound = " << upper_bound << " lower_bound = " << lower_bound << " greedy_gap = " << greedy_gap << " ratio = " << greedy_gap / initial_gap;
			lap::displayTime(start_time, ss.str().c_str(), lapDebug);
		}
#endif
		if (initial_gap < 4.0 * greedy_gap)
		{
			memcpy(v2, v, dim2 * sizeof(SC));
			// sort permutation by keys
			std::sort(perm, perm + dim, [&mod_v](int a, int b) { return (mod_v[a] > mod_v[b]) || ((mod_v[a] == mod_v[b]) && (a > b)); });

			lower_bound = 0.0;
			upper_bound = 0.0;
			// greedy search
			std::fill(mod_v, mod_v + dim2, SC(-1));
			for (int i = 0; i < dim2; i++)
			{
				// greedy order
				int j_min;
				SC min_cost, min_cost_real;
				if (i < dim)
				{
					const auto *tt = iterator.getRow(perm[i]);
					auto cost = [&tt, &v](int j) -> SC { return (SC)tt[j] - v[j]; };
					getMinimalCost(j_min, min_cost, min_cost_real, cost, mod_v, dim2);
				}
				else
				{
					auto cost = [&v](int j) -> SC { return -v[j]; };
					getMinimalCost(j_min, min_cost, min_cost_real, cost, mod_v, dim2);
				}
				upper_bound += min_cost + v[j_min];
				// need to use the same v values in total
				lower_bound += min_cost_real + v[j_min];
				mod_v[j_min] = SC(0);
				picked[i] = j_min;
			}
			greedy_gap = upper_bound - lower_bound;

#ifdef LAP_DEBUG
			{
				std::stringstream ss;
				ss << "upper_bound = " << upper_bound << " lower_bound = " << lower_bound << " greedy_gap = " << greedy_gap << " ratio = " << greedy_gap / initial_gap;
				lap::displayTime(start_time, ss.str().c_str(), lapDebug);
			}
#endif

			// update v in reverse order
			for (int i = dim2 - 1; i >= 0; --i)
			{
				if (perm[i] < dim)
				{
					const auto *tt = iterator.getRow(perm[i]);
					SC min_cost = (SC)tt[picked[i]] - v[picked[i]];
					mod_v[picked[i]] = SC(-1);
					for (int j = 0; j < dim2; j++)
					{
						if (mod_v[j] >= SC(0))
						{
							SC cost_l = (SC)tt[j] - v[j];
							if (cost_l < min_cost) v[j] -= min_cost - cost_l;
						}
					}
				}
				else
				{
					SC min_cost = -v[picked[i]];
					mod_v[picked[i]] = SC(-1);
					for (int j = 0; j < dim2; j++)
					{
						if (mod_v[j] >= SC(0))
						{
							SC cost_l = -v[j];
							if (cost_l < min_cost) v[j] -= min_cost - cost_l;
						}
					}
				}
			}

			normalizeV(v, dim2);

			double old_upper_bound = upper_bound;
			double old_lower_bound = lower_bound;
			upper_bound = 0.0;
			lower_bound = 0.0;
			for (int i = 0; i < dim2; i++)
			{
				SC min_cost, min_cost_real;
				if (perm[i] < dim)
				{
					const auto *tt = iterator.getRow(perm[i]);
					min_cost = (SC)tt[picked[i]];
					min_cost_real = std::numeric_limits<SC>::max();
					for (int j = 0; j < dim2; j++)
					{
						SC cost_l = (SC)tt[j] - v[j];
						min_cost_real = std::min(min_cost_real, cost_l);
					}
				}
				else
				{
					min_cost = SC(0);
					min_cost_real = std::numeric_limits<SC>::max();
					for (int j = 0; j < dim2; j++) min_cost_real = std::min(min_cost_real, -v[j]);
				}
				// need to use all picked v for the lower bound as well
				upper_bound += min_cost;
				lower_bound += min_cost_real + v[picked[i]];
			}
			upper_bound = std::min(upper_bound, old_upper_bound);
			lower_bound = std::max(lower_bound, old_lower_bound);
			greedy_gap = upper_bound - lower_bound;
			double ratio2 = greedy_gap / initial_greedy_gap;
#ifdef LAP_DEBUG
			{
				std::stringstream ss;
				ss << "upper_bound = " << upper_bound << " lower_bound = " << lower_bound << " greedy_gap = " << greedy_gap << " ratio = " << greedy_gap / initial_gap;
				lap::displayTime(start_time, ss.str().c_str(), lapDebug);
			}
#endif
			if (ratio2 > 1.0e-09)
			{
				for (int i = 0; i < dim2; i++)
				{
					v[i] = (SC)((double)v2[i] * ratio2 + (double)v[i] * (1.0 - ratio2));
				}
			}
		}

		SC upper, lower;
		getUpperLower(upper, lower, greedy_gap, initial_gap, dim, dim2);

		lapFree(mod_v);
		lapFree(picked);
		lapFree(v2);

		return std::pair<SC, SC>((SC)upper, (SC)lower);
	}

#if defined(__GNUC__)
#define __forceinline \
        __inline__ __attribute__((always_inline))
#endif

	__forceinline void dijkstraCheck(int& endofpath, bool& unassignedfound, int jmin, int* colsol, char* colactive)
	{
		colactive[jmin] = 0;
		if (colsol[jmin] < 0)
		{
			endofpath = jmin;
			unassignedfound = true;
		}
	}

	template <class SC>
	__forceinline void updateColumnPrices(char* colactive, int start, int end, SC min, SC* v, SC* d)
	{
		for (int j1 = start; j1 < end; j1++)
		{
			if (colactive[j1] == 0)
			{
				SC dlt = min - d[j1];
				v[j1] -= dlt;
			}
		}
	}

	template <class SC>
	__forceinline void updateColumnPrices(char* colactive, int start, int end, SC min, SC* v, SC* d, SC eps, SC& total, SC& total_eps)
	{
		for (int j1 = start; j1 < end; j1++)
		{
			if (colactive[j1] == 0)
			{
				SC dlt = min - d[j1];
				total += dlt;
				total_eps += eps;
				v[j1] -= dlt + eps;
			}
		}
	}

	__forceinline void resetRowColumnAssignment(int &endofpath, int f, int *pred, int *rowsol, int *colsol)
	{
		int i;
		do
		{
			i = pred[endofpath];
			colsol[endofpath] = i;
			std::swap(endofpath, rowsol[i]);
		} while (i != f);
	}

	template <class SC, class TC>
	void getNextEpsilon(TC &epsilon, TC &epsilon_lower, SC total_d, SC total_eps, bool first, bool second, int dim2)
	{
		if (epsilon > TC(0))
		{
			if (!first)
			{
				if ((TC(0.5) == TC(0)) && (epsilon == epsilon_lower)) epsilon = TC(0);
				else
				{
#ifdef LAP_DEBUG
					lapDebug << "  v_d = " << total_d / SC(dim2) << " v_eps = " << total_eps / SC(dim2) << " eps = " << epsilon;
#endif
					if ((!second) && (total_d > total_eps))
					{
						epsilon = TC(0);
					}
					else
					{
						epsilon = std::min(epsilon / TC(4), (TC)(total_eps / SC(8 * (size_t)dim2)));
					}
#ifdef LAP_DEBUG
					lapDebug << " -> " << epsilon;
#endif
					if (epsilon < epsilon_lower)
					{
						if (TC(0.5) == TC(0)) epsilon = epsilon_lower;
						else epsilon = TC(0);
					}
#ifdef LAP_DEBUG
					lapDebug << " -> " << epsilon << std::endl;
#endif
				}
			}
		}
	}

	template <class SC, class CF, class I>
	void solve(int dim, int dim2, CF &costfunc, I &iterator, int *rowsol, bool use_epsilon)

		// input:
		// dim        - problem size
		// costfunc - cost matrix
		// findcost   - searching cost matrix

		// output:
		// rowsol     - column assigned to row in solution
		// colsol     - row assigned to column in solution
		// u          - dual variables, row reduction numbers
		// v          - dual variables, column reduction numbers

	{
#ifndef LAP_QUIET
		auto start_time = std::chrono::high_resolution_clock::now();

		long long total_hit = 0LL;
		long long total_miss = 0LL;

		long long total_rows = 0LL;
		long long total_virtual = 0LL;

		long long last_rows = 0LL;
		long long last_virtual = 0LL;

		int elapsed = -1;
#else
#ifdef LAP_DISPLAY_EVALUATED
		long long total_hit = 0LL;
		long long total_miss = 0LL;

		long long total_rows = 0LL;
		long long total_virtual = 0LL;
#endif
#endif

		int  *pred;
		int  endofpath;
		char *colactive;
		SC *d;
		int *colsol;
		SC epsilon_upper;
		SC epsilon_lower;
		SC *v;
		int *perm;

#ifdef LAP_DEBUG
		std::vector<SC *> v_list;
		std::vector<SC> eps_list;
#endif

		lapAlloc(colactive, dim2, __FILE__, __LINE__);
		lapAlloc(d, dim2, __FILE__, __LINE__);
		lapAlloc(pred, dim2, __FILE__, __LINE__);
		lapAlloc(colsol, dim2, __FILE__, __LINE__);
		lapAlloc(v, dim2, __FILE__, __LINE__);
		lapAlloc(perm, dim2, __FILE__, __LINE__);

#ifdef LAP_ROWS_SCANNED
		unsigned long long *scancount;
		unsigned long long *pathlength;
		lapAlloc(scancount, dim2, __FILE__, __LINE__);
		lapAlloc(pathlength, dim2, __FILE__, __LINE__);
		memset(scancount, 0, dim2 * sizeof(unsigned long long));
		memset(pathlength, 0, dim2 * sizeof(unsigned long long));
#endif

		SC epsilon;

		if (use_epsilon)
		{
			std::pair<SC, SC> eps = estimateEpsilon(dim, dim2, iterator, v, perm);
			epsilon_upper = eps.first;
			epsilon_lower = eps.second;
		}
		else
		{
			memset(v, 0, dim2 * sizeof(SC));
			epsilon_upper = SC(0);
			epsilon_lower = SC(0);
		}
		epsilon = epsilon_upper;

		bool first = true;
		bool second = false;
		bool reverse = true;

		if ((!use_epsilon) || (epsilon > SC(0)))
		{
			for (int i = 0; i < dim2; i++) perm[i] = i;
			reverse = false;
		}

		SC total_d = SC(0);
		SC total_eps = SC(0);
		while (epsilon >= SC(0))
		{
#ifdef LAP_DEBUG
			if (first)
			{
				SC *vv;
				lapAlloc(vv, dim2, __FILE__, __LINE__);
				v_list.push_back(vv);
				eps_list.push_back(epsilon);
				memcpy(v_list.back(), v, sizeof(SC) * dim2);
			}
#endif
			getNextEpsilon(epsilon, epsilon_lower, total_d, total_eps, first, second, dim2);

			total_d = SC(0);
			total_eps = SC(0);
#ifndef LAP_QUIET
			{
				std::stringstream ss;
				ss << "eps = " << epsilon;
				const std::string tmp = ss.str();
				displayTime(start_time, tmp.c_str(), lapInfo);
			}
#endif
			// this is to ensure termination of the while statement
			if (epsilon == SC(0)) epsilon = SC(-1.0);
			memset(rowsol, -1, dim2 * sizeof(int));
			memset(colsol, -1, dim2 * sizeof(int));
			int jmin, jmin_n;
			SC min, min_n;
			bool unassignedfound;

#ifndef LAP_QUIET
			int old_complete = 0;
#endif

#ifdef LAP_MINIMIZE_V
			//int dim_limit = ((reverse) || (epsilon < SC(0))) ? dim2 : dim;
			int dim_limit = dim2;
#else
			int dim_limit = dim2;
#endif

			// AUGMENT SOLUTION for each free row.
#ifndef LAP_QUIET
			displayProgress(start_time, elapsed, 0, dim_limit, " rows");
#endif
			for (int fc = 0; fc < dim_limit; fc++)
			{
				int f = perm[((reverse) && (fc < dim)) ? (dim - 1 - fc) : fc];
#ifndef LAP_QUIET
				if (f < dim) total_rows++; else total_virtual++;
#else
#ifdef LAP_DISPLAY_EVALUATED
				if (f < dim) total_rows++; else total_virtual++;
#endif
#endif
#ifdef LAP_ROWS_SCANNED
				scancount[f]++;
#endif

				unassignedfound = false;

				// Dijkstra search
				min = std::numeric_limits<SC>::max();
				jmin = dim2;
				if (f < dim)
				{
					auto tt = iterator.getRow(f);
					for (int j = 0; j < dim2; j++)
					{
						colactive[j] = 1;
						pred[j] = f;
						SC h = d[j] = tt[j] - v[j];
						if (h <= min)
						{
							if (h < min)
							{
								// better
								jmin = j;
								min = h;
							}
							else //if (h == min)
							{
								// same, do only update if old was used and new is free
								if ((colsol[jmin] >= 0) && (colsol[j] < 0)) jmin = j;
							}
						}
					}
				}
				else
				{
					for (int j = 0; j < dim2; j++)
					{
						colactive[j] = 1;
						pred[j] = f;
						SC h = d[j] = -v[j];
						if (colsol[j] < dim)
						{
							if (h <= min)
							{
								if (h < min)
								{
									// better
									jmin = j;
									min = h;
								}
								else //if (h == min)
								{
									// same, do only update if old was used and new is free
									if ((colsol[jmin] >= 0) && (colsol[j] < 0)) jmin = j;
								}
							}
						}
					}
				}

				dijkstraCheck(endofpath, unassignedfound, jmin, colsol, colactive);
				// marked skipped columns that were cheaper
				if (f >= dim)
				{
					for (int j = 0; j < dim2; j++)
					{
						// ignore any columns assigned to virtual rows
						if ((colsol[j] >= dim) && (d[j] <= min))
						{
							colactive[j] = 0;
						}
					}
				}

				while (!unassignedfound)
				{
					// update 'distances' between freerow and all unscanned columns, via next scanned column.
					int i = colsol[jmin];
#ifndef LAP_QUIET
					if (i < dim) total_rows++; else total_virtual++;
#else
#ifdef LAP_DISPLAY_EVALUATED
					if (i < dim) total_rows++; else total_virtual++;
#endif
#endif
#ifdef LAP_ROWS_SCANNED
					scancount[i]++;
#endif

					jmin_n = dim2;
					min_n = std::numeric_limits<SC>::max();
					if (i < dim)
					{
						auto tt = iterator.getRow(i);
						SC tt_jmin = (SC)tt[jmin];
						SC v_jmin = v[jmin];
						for (int j = 0; j < dim2; j++)
						{
							if (colactive[j] != 0)
							{
								SC v2 = (tt[j] - tt_jmin) - (v[j] - v_jmin) + min;
								SC h = d[j];
								if (v2 < h)
								{
									pred[j] = i;
									d[j] = v2;
									h = v2;
								}
								if (h <= min_n)
								{
									if (h < min_n)
									{
										// better
										jmin_n = j;
										min_n = h;
									}
									else //if (h == min_n)
									{
										// same, do only update if old was used and new is free
										if ((colsol[jmin_n] >= 0) && (colsol[j] < 0)) jmin_n = j;
									}
								}
							}
						}
					}
					else
					{
						SC v_jmin = v[jmin];
						for (int j = 0; j < dim2; j++)
						{
							if (colactive[j] != 0)
							{
								SC v2 = -(v[j] - v_jmin) + min;
								SC h = d[j];
								if (v2 < h)
								{
									pred[j] = i;
									d[j] = v2;
									h = v2;
								}
								if (h <= min_n)
								{
									if (colsol[j] < dim)
									{
										if (h < min_n)
										{
											// better
											jmin_n = j;
											min_n = h;
										}
										else //if (h == min_n)
										{
											// same, do only update if old was used and new is free
											if ((colsol[jmin_n] >= 0) && (colsol[j] < 0)) jmin_n = j;
										}
									}
								}
							}
						}
					}

					min = std::max(min, min_n);
					jmin = jmin_n;
					dijkstraCheck(endofpath, unassignedfound, jmin, colsol, colactive);
					// marked skipped columns that were cheaper
					if (i >= dim)
					{
						for (int j = 0; j < dim2; j++)
						{
							// ignore any columns assigned to virtual rows
							if ((colactive[j] == 1) && (colsol[j] >= dim) && (d[j] <= min))
							{
								colactive[j] = 0;
							}
						}
					}
				}

				// update column prices. can increase or decrease
				if (epsilon > SC(0))
				{
					updateColumnPrices(colactive, 0, dim2, min, v, d, epsilon, total_d, total_eps);
				}
				else
				{
					updateColumnPrices(colactive, 0, dim2, min, v, d);
				}
#ifdef LAP_ROWS_SCANNED
				{
					int i;
					int eop = endofpath;
					do
					{
						i = pred[eop];
						eop = rowsol[i];
						if (i != f) pathlength[f]++;
					} while (i != f);
				}
#endif

				// reset row and column assignments along the alternating path.
				resetRowColumnAssignment(endofpath, f, pred, rowsol, colsol);
#ifndef LAP_QUIET
				int level;
				if ((level = displayProgress(start_time, elapsed, fc + 1, dim_limit, " rows")) != 0)
				{
					long long hit, miss;
					iterator.getHitMiss(hit, miss);
					total_hit += hit;
					total_miss += miss;
					if ((hit != 0) || (miss != 0))
					{
						if (level == 1) lapInfo << "  hit: " << hit << " miss: " << miss << " (" << miss - (f + 1 - old_complete) << " + " << f + 1 - old_complete << ")" << std::endl;
						else lapDebug << "  hit: " << hit << " miss: " << miss << " (" << miss - (f + 1 - old_complete) << " + " << f + 1 - old_complete << ")" << std::endl;
					}
					old_complete = f + 1;
				}
#endif
			}

#ifdef LAP_MINIMIZE_V
			if (epsilon > SC(0))
			{
#if 0
				if (dim_limit < dim2) normalizeV(v, dim2, colsol);
				else normalizeV(v, dim2);
#else
				if (dim_limit < dim2) for (int i = 0; i < dim2; i++) if (colsol[i] < 0) v[i] -= SC(2) * epsilon;
				normalizeV(v, dim2);
#endif
			}
#endif

#ifdef LAP_DEBUG
			if (epsilon > SC(0))
			{
				SC *vv;
				lapAlloc(vv, dim2, __FILE__, __LINE__);
				v_list.push_back(vv);
				eps_list.push_back(epsilon);
				memcpy(v_list.back(), v, sizeof(SC) * dim2);
			}
			else
			{
				int count = (int)v_list.size();
				if (count > 0)
				{
					for (int l = 0; l < count; l++)
					{
						SC dlt(0), dlt2(0);
						for (int i = 0; i < dim2; i++)
						{
							SC diff = v_list[l][i] - v[i];
							dlt += diff;
							dlt2 += diff * diff;
						}
						dlt /= SC(dim2);
						dlt2 /= SC(dim2);
						lapDebug << "iteration = " << l << " eps/mse = " << eps_list[l] << " " << dlt2 - dlt * dlt << " eps/rmse = " << eps_list[l] << " " << sqrt(dlt2 - dlt * dlt) << std::endl;
						lapFree(v_list[l]);
					}
				}
			}
#endif
			second = first;
			first = false;
			reverse = !reverse;
#ifndef LAP_QUIET
			lapInfo << "  rows evaluated: " << total_rows;
			if (last_rows > 0LL) lapInfo << " (+" << total_rows - last_rows << ")";
			last_rows = total_rows;
			if (total_virtual > 0) lapInfo << " virtual rows evaluated: " << total_virtual;
			if (last_virtual > 0LL) lapInfo << " (+" << total_virtual - last_virtual << ")";
			last_virtual = total_virtual;
			lapInfo << std::endl;
			if ((total_hit != 0) || (total_miss != 0)) lapInfo << "  hit: " << total_hit << " miss: " << total_miss << std::endl;
#endif
		}

#ifdef LAP_QUIET
#ifdef LAP_DISPLAY_EVALUATED
		iterator.getHitMiss(total_hit, total_miss);
		lapInfo << "  rows evaluated: " << total_rows;
		if (total_virtual > 0) lapInfo << " virtual rows evaluated: " << total_virtual;
		lapInfo << std::endl;
		if ((total_hit != 0) || (total_miss != 0)) lapInfo << "  hit: " << total_hit << " miss: " << total_miss << std::endl;
#endif
#endif

#ifdef LAP_ROWS_SCANNED
		lapInfo << "row\tscanned\tlength" << std::endl;
		for (int f = 0; f < dim2; f++)
		{
			lapInfo << f << "\t" << scancount[f] << "\t" << pathlength[f] << std::endl;
		}

		lapFree(scancount);
		lapFree(pathlength);
#endif

#ifdef LAP_VERIFY_RESULT
		SC slack = SC(0);
		bool correct = true;
		for (int f = 0; f < dim2; f++)
		{
			auto tt = iterator.getRow(f);
			int jmin = rowsol[f];
			SC ref_min = tt[jmin] - v[jmin];
			SC min = ref_min;
			for (int j = 0; j < dim2; j++)
			{
				SC h = tt[j] - v[j];
				if (h < min)
				{
					// better
					jmin = j;
					min = h;
				}
			}
			if (jmin != rowsol[f])
			{
				slack += ref_min - min;
				correct = false;
			}
		}
		if (correct)
		{
			lapInfo << "Solution accurate." << std::endl;
		}
		else
		{
			lapInfo << "Solution might be inaccurate (slack = " << slack << ")." << std::endl;
		}
#endif

		// free reserved memory.
		lapFree(pred);
		lapFree(colactive);
		lapFree(d);
		lapFree(v);
		lapFree(colsol);
		lapFree(perm);
	}

	// shortcut for square problems
	template <class SC, class CF, class I>
	void solve(int dim, CF &costfunc, I &iterator, int *rowsol, bool use_epsilon)
	{
		solve<SC>(dim, dim, costfunc, iterator, rowsol, use_epsilon);
	}

	template <class SC, class CF>
	SC cost(int dim, int dim2, CF &costfunc, int *rowsol)
	{
		SC total = SC(0);
		for (int i = 0; i < dim; i++) total += costfunc.getCost(i, rowsol[i]);
		return total;
	}

	template <class SC, class CF>
	SC cost(int dim, CF &costfunc, int *rowsol)
	{
		return cost<SC, CF>(dim, dim, costfunc, rowsol);
	}
}
