#pragma once

#include "../lap_solver.h"

namespace lap
{
	namespace omp
	{
		template <class SC, class I>
		std::pair<SC, SC> estimateEpsilon(int dim, int dim2, I& iterator, SC *v, int *perm)
		{
#ifdef LAP_DEBUG
			auto start_time = std::chrono::high_resolution_clock::now();
#endif
			SC *mod_v;
			int *picked;
			SC *merge_cost;
			int *merge_idx;
			SC *v2;
			int threads = omp_get_max_threads();

			lapAlloc(mod_v, dim2, __FILE__, __LINE__);
			lapAlloc(v2, dim2, __FILE__, __LINE__);
			lapAlloc(picked, dim2, __FILE__, __LINE__);
			lapAlloc(merge_cost, std::max((long long)(threads + 1) * (long long)dim2, (long long)threads << 5), __FILE__, __LINE__);
			lapAlloc(merge_idx, (long long)threads << 5, __FILE__, __LINE__);

			double lower_bound = 0.0;
			double upper_bound = 0.0;
			double greedy_bound = 0.0;

			memset(picked, 0, sizeof(int) * dim2);

#pragma omp parallel
			{
				int t = omp_get_thread_num();
				for (int i = 0; i < dim2; i++)
				{
					int off = ((i & 1) == 0) ? 0 : (threads << 2);
					SC min_cost_l, max_cost_l, picked_cost_l;
					int j_min = dim2;
					if (i < dim)
					{
						const auto *tt = iterator.getRow(t, i);
						auto cost = [&tt](int j) -> SC { return (SC)tt[j]; };
						getMinMaxBest(i - iterator.ws.part[t].first, min_cost_l, max_cost_l, picked_cost_l, j_min, cost, picked + iterator.ws.part[t].first, iterator.ws.part[t].second - iterator.ws.part[t].first);
						if (j_min >= dim2) j_min = dim2; else j_min += iterator.ws.part[t].first;
						// a little hacky
						if ((i >= iterator.ws.part[t].first) && (i < iterator.ws.part[t].second))
						{
							merge_cost[off + (threads << 1)] = max_cost_l;
						}
						merge_cost[off + t] = min_cost_l;
						merge_cost[off + t + threads] = picked_cost_l;
						merge_idx[off + t] = j_min;
#pragma omp barrier
						min_cost_l = merge_cost[off];
						picked_cost_l = merge_cost[off + threads];
						max_cost_l = merge_cost[off + (threads << 1)];
						j_min = merge_idx[off];
						for (int ii = 1; ii < threads; ii++)
						{
							min_cost_l = std::min(min_cost_l, merge_cost[off + ii]);
							if (merge_cost[off + ii + threads] < picked_cost_l)
							{
								picked_cost_l = merge_cost[off + ii + threads];
								j_min = merge_idx[off + ii];
							}
						}
						if ((j_min >= iterator.ws.part[t].first) && (j_min < iterator.ws.part[t].second)) picked[j_min] = 1;
						updateEstimatedV(v + iterator.ws.part[t].first, mod_v + iterator.ws.part[t].first, cost, (i == 0), (i == 1), min_cost_l, max_cost_l, iterator.ws.part[t].second - iterator.ws.part[t].first);
					}
					else
					{
						auto cost = [](int j) -> SC { return SC(0); };
						getMinMaxBest(i - iterator.ws.part[t].first, min_cost_l, max_cost_l, picked_cost_l, j_min, cost, picked + iterator.ws.part[t].first, iterator.ws.part[t].second - iterator.ws.part[t].first);
						if (j_min >= dim2) j_min = dim2; else j_min += iterator.ws.part[t].first;
						// a little hacky
						if ((i >= iterator.ws.part[t].first) && (i < iterator.ws.part[t].second))
						{
							merge_cost[off + (threads << 1)] = max_cost_l;
						}
						merge_cost[off + t] = min_cost_l;
						merge_cost[off + t + threads] = picked_cost_l;
						merge_idx[off + t] = j_min;
#pragma omp barrier
						min_cost_l = merge_cost[off];
						picked_cost_l = merge_cost[off + threads];
						max_cost_l = merge_cost[off + (threads << 1)];
						j_min = merge_idx[off];
						for (int ii = 1; ii < threads; ii++)
						{
							min_cost_l = std::min(min_cost_l, merge_cost[off + ii]);
							if (merge_cost[off + ii + threads] < picked_cost_l)
							{
								picked_cost_l = merge_cost[off + ii + threads];
								j_min = merge_idx[off + ii];
							}
						}
						if ((j_min >= iterator.ws.part[t].first) && (j_min < iterator.ws.part[t].second)) picked[j_min] = 1;
						updateEstimatedV(v + iterator.ws.part[t].first, mod_v + iterator.ws.part[t].first, cost, (i == 0), (i == 1), min_cost_l, max_cost_l, iterator.ws.part[t].second - iterator.ws.part[t].first);
					}
					if (t == 0)
					{
						picked[j_min] = 1;
						lower_bound += min_cost_l;
						upper_bound += max_cost_l;
						greedy_bound += picked_cost_l;
					}
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

#pragma omp parallel
			{
				int t = omp_get_thread_num();
				// reverse order
				for (int i = dim2 - 1; i >= 0; --i)
				{
					int off = ((i & 1) == 0) ? 0 : (threads << 2);
					SC min_cost_l, second_cost_l, picked_cost_l;
					int j_min = dim2;
					if (i < dim)
					{
						const auto *tt = iterator.getRow(t, i);
						auto cost = [&tt, &v, &iterator, &t](int j) -> SC { return (SC)tt[j] - v[j + iterator.ws.part[t].first]; };
						getMinSecondBest(min_cost_l, second_cost_l, picked_cost_l, j_min, cost, picked + iterator.ws.part[t].first, iterator.ws.part[t].second - iterator.ws.part[t].first);
					}
					else
					{
						auto cost = [&v, &iterator, &t](int j) -> SC { return -v[j + iterator.ws.part[t].first]; };
						getMinSecondBest(min_cost_l, second_cost_l, picked_cost_l, j_min, cost, picked + iterator.ws.part[t].first, iterator.ws.part[t].second - iterator.ws.part[t].first);
					}
					if (j_min >= dim2) j_min = dim2; else j_min += iterator.ws.part[t].first;
					merge_cost[off + t] = min_cost_l;
					merge_cost[off + t + threads] = second_cost_l;
					merge_cost[off + t + (threads << 1)] = picked_cost_l;
					merge_cost[off + t + (threads * 3)] = (j_min < dim2)?v[j_min]:std::numeric_limits<SC>::max();
					merge_idx[off + t] = j_min;
#pragma omp barrier
					min_cost_l = merge_cost[off];
					second_cost_l = merge_cost[off + threads];
					picked_cost_l = merge_cost[off + (threads << 1)];
					SC v_jmin = merge_cost[off + threads * 3];
					j_min = merge_idx[off];
					for (int ii = 1; ii < threads; ii++)
					{
						if (merge_cost[off + ii] < min_cost_l)
						{
							second_cost_l = std::min(min_cost_l, merge_cost[off + ii + threads]);
							min_cost_l = merge_cost[off + ii];
						}
						else
						{
							second_cost_l = std::min(second_cost_l, merge_cost[off + ii]);
						}
						if (merge_cost[off + ii + (threads << 1)] < picked_cost_l)
						{
							picked_cost_l = merge_cost[off + ii + (threads << 1)];
							j_min = merge_idx[off + ii];
							v_jmin = merge_cost[off + ii + (threads * 3)];
						}
					}
					if ((j_min >= iterator.ws.part[t].first) && (j_min < iterator.ws.part[t].second)) picked[j_min] = 1;
					if (t == 0)
					{
						perm[i] = i;
						mod_v[i] = second_cost_l - min_cost_l;
						// need to use the same v values in total
						lower_bound += min_cost_l + v_jmin;
						upper_bound += picked_cost_l + v_jmin;
					}
				}
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
#pragma omp parallel
				{
					int t = omp_get_thread_num();
					for (int i = 0; i < dim2; i++)
					{
						int off = ((i & 1) == 0) ? 0 : (threads << 2);
						// greedy order
						int j_min = dim2;
						SC min_cost, min_cost_real;
						if (perm[i] < dim)
						{
							const auto* tt = iterator.getRow(t, perm[i]);
							auto cost = [&tt, &v, &iterator, &t](int j) -> SC { return (SC)tt[j] - v[j + iterator.ws.part[t].first]; };
							getMinimalCost(j_min, min_cost, min_cost_real, cost, mod_v + iterator.ws.part[t].first, iterator.ws.part[t].second - iterator.ws.part[t].first);
						}
						else
						{
							auto cost = [&v, &iterator, &t](int j) -> SC { return -v[j + iterator.ws.part[t].first]; };
							getMinimalCost(j_min, min_cost, min_cost_real, cost, mod_v + iterator.ws.part[t].first, iterator.ws.part[t].second - iterator.ws.part[t].first);
						}
						if (j_min >= dim2) j_min = dim2; else j_min += iterator.ws.part[t].first;
						merge_cost[off + t] = min_cost;
						merge_cost[off + t + threads] = min_cost_real;
						merge_cost[off + t + (threads << 1)] = (j_min < dim2) ? v[j_min] : std::numeric_limits<SC>::max();;
						merge_idx[off + t] = j_min;
#pragma omp barrier
						min_cost = merge_cost[off];
						min_cost_real = merge_cost[off + threads];
						SC v_jmin = merge_cost[off + (threads << 1)];
						j_min = merge_idx[off];
						for (int ii = 1; ii < threads; ii++)
						{
							if (merge_cost[off + ii] < min_cost)
							{
								min_cost = merge_cost[off + ii];
								j_min = merge_idx[off + ii];
								v_jmin = merge_cost[off + ii + (threads << 1)];
							}
							min_cost_real = std::min(min_cost_real, merge_cost[off + ii + threads]);
						}
						if ((j_min >= iterator.ws.part[t].first) && (j_min < iterator.ws.part[t].second))
						{
							mod_v[j_min] = SC(0);
							picked[i] = j_min;
						}
						if (t == 0)
						{
							upper_bound += min_cost + v_jmin;
							// need to use the same v values in total
							lower_bound += min_cost_real + v_jmin;
						}
					}
				}

				greedy_gap = upper_bound - lower_bound;

#ifdef LAP_DEBUG
				{
					std::stringstream ss;
					ss << "upper_bound = " << upper_bound << " lower_bound = " << lower_bound << " greedy_gap = " << greedy_gap << " ratio = " << greedy_gap / initial_gap;
					lap::displayTime(start_time, ss.str().c_str(), lapDebug);
				}
#endif

#pragma omp parallel
				{
					int t = omp_get_thread_num();
					// update v in reverse order
					for (int i = dim2 - 1; i >= 0; --i)
					{
						int off = ((i & 1) == 0) ? 0 : 32;
						if (perm[i] < dim)
						{
							const auto *tt = iterator.getRow(t, perm[i]);
							if ((picked[i] >= iterator.ws.part[t].first) && (picked[i] < iterator.ws.part[t].second))
							{
								merge_cost[off] = (SC)tt[picked[i] - iterator.ws.part[t].first] - v[picked[i]];
								mod_v[picked[i]] = SC(-1);
							}
#pragma omp barrier
							SC min_cost = merge_cost[off];
							for (int j = iterator.ws.part[t].first; j < iterator.ws.part[t].second; j++)
							{
								if (mod_v[j] >= SC(0))
								{
									SC cost_l = (SC)tt[j - iterator.ws.part[t].first] - v[j];
									if (cost_l < min_cost) v[j] -= min_cost - cost_l;
								}
							}
						}
						else
						{
							if ((picked[i] >= iterator.ws.part[t].first) && (picked[i] < iterator.ws.part[t].second))
							{
								merge_cost[off] = -v[picked[i]];
								mod_v[picked[i]] = SC(-1);
							}
#pragma omp barrier
							SC min_cost = merge_cost[off];
							for (int j = iterator.ws.part[t].first; j < iterator.ws.part[t].second; j++)
							{
								if (mod_v[j] >= SC(0))
								{
									SC cost_l = -v[j];
									if (cost_l < min_cost) v[j] -= min_cost - cost_l;
								}
							}
						}
					}
				}

				normalizeV(v, dim2);

				double old_upper_bound = upper_bound;
				double old_lower_bound = lower_bound;
				upper_bound = 0.0;
				lower_bound = 0.0;
				int off = threads * dim2;
#pragma omp parallel
				{
					int t = omp_get_thread_num();
					for (int i = 0; i < dim2; i++)
					{
						SC min_cost_real;
						if (perm[i] < dim)
						{
							const auto* tt = iterator.getRow(t, perm[i]);
							if ((picked[i] >= iterator.ws.part[t].first) && (picked[i] < iterator.ws.part[t].second))
							{
								merge_cost[i + off] = (SC)tt[picked[i] - iterator.ws.part[t].first];
							}
							min_cost_real = std::numeric_limits<SC>::max();

							for (int j = iterator.ws.part[t].first; j < iterator.ws.part[t].second; j++)
							{
								SC cost_l = (SC)tt[j - iterator.ws.part[t].first] - v[j];
								min_cost_real = std::min(min_cost_real, cost_l);
							}
						}
						else
						{
							min_cost_real = std::numeric_limits<SC>::max();
							if (t == 0) merge_cost[i + off] = SC(0);
							for (int j = iterator.ws.part[t].first; j < iterator.ws.part[t].second; j++) min_cost_real = std::min(min_cost_real, -v[j]);
						}
						merge_cost[i + t * dim2] = min_cost_real;
					}
				}
				for (int i = 0; i < dim2; i++)
				{
					SC min_cost_real = merge_cost[i];
					for (int ii = 1; ii < threads; ii++) min_cost_real = std::min(min_cost_real, merge_cost[i + ii * dim2]);
					lower_bound += min_cost_real + v[picked[i]];
					upper_bound += merge_cost[i + off];
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
			lapFree(merge_cost);
			lapFree(merge_idx);
			lapFree(v2);

			return std::pair<SC, SC>((SC)upper, (SC)lower);
		}

		__forceinline void dijkstraCheck(int& endofpath, bool& unassignedfound, int jmin, int* colsol, char** colactive, int t, std::pair<int, int>* part)
		{
			if ((jmin >= part[t].first) && (jmin < part[t].second)) colactive[t][jmin - part[t].first] = 0;
			if (colsol[jmin] < 0)
			{
				endofpath = jmin;
				unassignedfound = true;
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

			int *pred;
			int endofpath;
			char **colactive;
			SC **d;
			int *colsol;
			SC epsilon_upper;
			SC epsilon_lower;
			SC **v;
			int *perm;
			decltype(iterator.getRow(0, 0))* tt;

			SC v_jmin_global;
			SC tt_jmin_global;

#ifdef LAP_DEBUG
			std::vector<SC *> v_list;
			std::vector<SC> eps_list;
#endif

			lapAlloc(pred, dim2, __FILE__, __LINE__);
			lapAlloc(colsol, dim2, __FILE__, __LINE__);
			lapAlloc(colactive, omp_get_max_threads(), __FILE__, __LINE__);
			lapAlloc(d, omp_get_max_threads(), __FILE__, __LINE__);
			lapAlloc(v, omp_get_max_threads(), __FILE__, __LINE__);
			lapAlloc(perm, dim2, __FILE__, __LINE__);
			lapAlloc(tt, omp_get_max_threads(), __FILE__, __LINE__);


#pragma omp parallel
			{
				int t = omp_get_thread_num();
				int start = iterator.ws.part[t].first;
				int end = iterator.ws.part[t].second;
				int count = end - start;

				lapAlloc(colactive[t], count, __FILE__, __LINE__);
				lapAlloc(d[t], count, __FILE__, __LINE__);
				lapAlloc(v[t], count, __FILE__, __LINE__);
			}

			SC *min_private;
			int *jmin_private;
			// use << 4 to avoid false sharing
			lapAlloc(min_private, omp_get_max_threads() << 4, __FILE__, __LINE__);
			lapAlloc(jmin_private, omp_get_max_threads() << 4, __FILE__, __LINE__);

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
				SC* tmp_v;
				lapAlloc(tmp_v, dim2, __FILE__, __LINE__);
				std::pair<SC, SC> eps = lap::omp::estimateEpsilon(dim, dim2, iterator, tmp_v, perm);
#pragma omp parallel
				{
					int t = omp_get_thread_num();
					int start = iterator.ws.part[t].first;
					int end = iterator.ws.part[t].second;
					int count = end - start;
					memcpy(v[t], &(tmp_v[start]), count * sizeof(SC));
				}
				lapFree(tmp_v);
				epsilon_upper = eps.first;
				epsilon_lower = eps.second;
			}
			else
			{
#pragma omp parallel
				{
					int t = omp_get_thread_num();
					int start = iterator.ws.part[t].first;
					int end = iterator.ws.part[t].second;
					int count = end - start;
					memset(v[t], 0, count * sizeof(SC));
				}
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
				lap::getNextEpsilon(epsilon, epsilon_lower, total_d, total_eps, first, second, dim2);
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

#ifndef LAP_QUIET
				int old_complete = 0;
#endif

#ifdef LAP_MINIMIZE_V
				int dim_limit = dim2;
#else
				int dim_limit = dim2;
#endif

				// AUGMENT SOLUTION for each free row.
#ifndef LAP_QUIET
				displayProgress(start_time, elapsed, 0, dim2, " rows");
#endif

#pragma omp parallel
				{
					const int t = omp_get_thread_num();
					const int start = iterator.ws.part[t].first;
					const int end = iterator.ws.part[t].second;
					const int count = end - start;

					char* colactive_private = colactive[t];
					SC* d_private = d[t];
					int* pred_private = &(pred[start]);
					SC* v_private = v[t];
					int* colsol_private = &(colsol[start]);

					memset(colsol_private, -1, count * sizeof(int));

					int threads = omp_get_num_threads();

					for (int fc = 0; fc < dim_limit; fc++)
					{
						int f = perm[((reverse) && (fc < dim)) ? (dim - 1 - fc) : fc];
						int jmin_local = dim2;
						SC min_local = std::numeric_limits<SC>::max();
						bool unassignedfound_local;
						if (f < dim)
						{
							tt[t] = iterator.getRow(t, f);
							for (int j = 0; j < count; j++)
							{
								colactive_private[j] = 1;
								pred_private[j] = f;
								SC h = d_private[j] = tt[t][j] - v_private[j];
								if (h < min_local)
								{
									// better
									jmin_local = j;
									min_local = h;
								}
								// same, do only update if old was used and new is free
								else if ((h == min_local) && (colsol_private[jmin_local] >= 0) && (colsol_private[j] < 0)) jmin_local = j;
							}
						}
						else
						{
							min_local = std::numeric_limits<SC>::max();
							for (int j = 0; j < count; j++)
							{
								colactive_private[j] = 1;
								pred_private[j] = f;
								SC h = d_private[j] = -v_private[j];
								if (h < min_local)
								{
									// ignore any columns assigned to virtual rows
									if (colsol_private[j] < dim)
									{
										// better
										jmin_local = j;
										min_local = h;
									}
								}
								// same, do only update if old was used and new is free
								else if ((h == min_local) && (colsol_private[jmin_local] >= 0) && (colsol_private[j] < 0)) jmin_local = j;
							}
						}
						min_private[t] = min_local;
						jmin_private[t] = jmin_local + start;
#pragma omp barrier
#ifndef LAP_QUIET
						if (t == 0) { if (f < dim) total_rows++; else total_virtual++; }
#else
#ifdef LAP_DISPLAY_EVALUATED
						if (t == 0) { if (f < dim) total_rows++; else total_virtual++; }
#endif
#endif
#ifdef LAP_ROWS_SCANNED
						if (t == 0) scancount[f]++;
#endif
						bool taken = false;
						min_local = min_private[0];
						jmin_local = jmin_private[0];
						for (int ii = 1; ii < omp_get_num_threads(); ii++)
						{
							if (min_private[ii] < min_local)
							{
								// better than previous
								min_local = min_private[ii];
								jmin_local = jmin_private[ii];
								taken = (colsol[jmin_local] >= 0);
							}
							else if ((min_private[ii] == min_local) && (taken) && (colsol[jmin_private[ii]] < 0))
							{
								jmin_local = jmin_private[ii];
								taken = false;
							}
						}
						unassignedfound_local = false;
						dijkstraCheck(endofpath, unassignedfound_local, jmin_local, colsol, colactive, t, iterator.ws.part);

						// marked skipped columns that were cheaper
						if (f >= dim)
						{
							for (int j = 0; j < count; j++)
							{
								// ignore any columns assigned to virtual rows
								if ((colsol_private[j] >= dim) && (d_private[j] <= min_local))
								{
									colactive_private[j] = 0;
								}
							}
						}
						while (!unassignedfound_local)
						{
							// update 'distances' between freerow and all unscanned columns, via next scanned column.
							int i = colsol[jmin_local];
							SC min_n_local = std::numeric_limits<SC>::max();
							if (i < dim)
							{
								tt[t] = iterator.getRow(t, i);
								if ((jmin_local >= start) && (jmin_local < end))
								{
									v_jmin_global = v[t][jmin_local - start];
									tt_jmin_global = (SC)tt[t][jmin_local - start];
								}
								jmin_local = dim2;
#pragma omp barrier
								SC v_jmin = v_jmin_global;
								SC tt_jmin = tt_jmin_global;
								for (int j = 0; j < count; j++)
								{
									if (colactive_private[j] != 0)
									{
										SC v2 = (tt[t][j] - tt_jmin) - (v_private[j] - v_jmin) + min_local;
										SC h = d_private[j];
										if (v2 < h)
										{
											pred_private[j] = i;
											d_private[j] = v2;
											h = v2;
										}
										if (h < min_n_local)
										{
											// better
											jmin_local = j;
											min_n_local = h;
										}
										// same, do only update if old was used and new is free
										else if ((h == min_n_local) && (colsol_private[jmin_local] >= 0) && (colsol_private[j] < 0)) jmin_local = j;
									}
								}
							}
							else
							{
								if ((jmin_local >= start) && (jmin_local < end))
								{
									v_jmin_global = v[t][jmin_local - start];
								}
								jmin_local = dim2;
#pragma omp barrier
								SC v_jmin = v_jmin_global;
								for (int j = 0; j < count; j++)
								{
									if (colactive_private[j] != 0)
									{
										SC v2 = -(v_private[j] - v_jmin) + min_local;
										SC h = d_private[j];
										if (v2 < h)
										{
											pred_private[j] = i;
											d_private[j] = v2;
											h = v2;
										}
										if (h < min_n_local)
										{
											// ignore any columns assigned to virtual rows
											if (colsol_private[j] < dim)
											{
												// better
												jmin_local = j;
												min_n_local = h;
											}
										}
										// same, do only update if old was used and new is free
										else if ((h == min_n_local) && (colsol_private[jmin_local] >= 0) && (colsol_private[j] < 0)) jmin_local = j;
									}
								}
							}
							min_private[t] = min_n_local;
							jmin_private[t] = jmin_local + start;
#pragma omp barrier
#ifndef LAP_QUIET
							if (t == 0) { if (f < dim) total_rows++; else total_virtual++; }
#else
#ifdef LAP_DISPLAY_EVALUATED
							if (t == 0) { if (f < dim) total_rows++; else total_virtual++; }
#endif
#endif
#ifdef LAP_ROWS_SCANNED
							if (t == 0) scancount[i]++;
#endif
							bool taken = false;
							min_n_local = min_private[0];
							jmin_local = jmin_private[0];
							for (int ii = 1; ii < omp_get_num_threads(); ii++)
								{
								if (min_private[ii] < min_n_local)
								{
									// better than previous
									min_n_local = min_private[ii];
									jmin_local = jmin_private[ii];
									taken = (colsol[jmin_local] >= 0);
								}
								else if ((min_private[ii] == min_n_local) && (taken) && (colsol[jmin_private[ii]] < 0))
								{
									jmin_local = jmin_private[ii];
									taken = false;
								}
							}
							unassignedfound_local = false;
							dijkstraCheck(endofpath, unassignedfound_local, jmin_local, colsol, colactive, t, iterator.ws.part);
							min_local = std::max(min_n_local, min_local);
							// marked skipped columns that were cheaper
							if (i >= dim)
							{
								for (int j = 0; j < count; j++)
								{
									// ignore any columns assigned to virtual rows
									if ((colactive_private[j] == 1) && (colsol_private[j] >= dim) && (d_private[j] <= min_local))
									{
										colactive_private[j] = 0;
									}
								}
							}
						}
						// update column prices. can increase or decrease
						if (epsilon > SC(0))
						{
							min_private[t + 2 * threads] = SC(0);
							min_private[t + 3 * threads] = SC(0);
							updateColumnPrices(colactive_private, 0, count, min_local, v_private, d_private, epsilon, min_private[t + 2 * threads], min_private[t + 3 * threads]);
						}
						else
						{
							updateColumnPrices(colactive_private, 0, count, min_local, v_private, d_private);
						}
#pragma omp barrier
						if (t == 0)
						{
							if (epsilon > SC(0)) for (int tt = 0; tt < omp_get_num_threads(); tt++)
							{
								total_d += min_private[tt + 2 * threads];
								total_eps += min_private[tt + 3 * threads];
							}
						}
#ifdef LAP_ROWS_SCANNED
						if (t == 1)
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
						if (t == 1)
						{
							// reset row and column assignments along the alternating path.
							lap::resetRowColumnAssignment(endofpath, f, pred, rowsol, colsol);
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
									{
										if (level == 1) lapInfo << "  hit: " << hit << " miss: " << miss << " (" << miss - (f + 1 - old_complete) << " + " << f + 1 - old_complete << ")" << std::endl;
										else lapDebug << "  hit: " << hit << " miss: " << miss << " (" << miss - (f + 1 - old_complete) << " + " << f + 1 - old_complete << ")" << std::endl;
									}
								}
								old_complete = f + 1;
							}
#endif
						}
#pragma omp barrier
					}
				}

#ifdef LAP_MINIMIZE_V
				if (epsilon > SC(0))
				{
#if 0
					if (dim_limit < dim2) normalizeV(v, dim2, colsol);
					else lap::normalizeV(v, dim2);
#else
					if (dim_limit < dim2)
					{
						for (int j = 0; j < omp_get_max_threads(); j++)
						{
							for (int i = 0; i < iterator.ws.part[j].second - iterator.ws.part[j].first; i++)
							{
								if (colsol[i + iterator.ws.part[j].first] < 0) v[j][i] -= SC(2) * epsilon;
							}
						}
					}
					SC max_v = v[0][0];
					for (int j = 0; j < omp_get_max_threads(); j++)
					{
						for (int i = 0; i < iterator.ws.part[j].second - iterator.ws.part[j].first; i++)
						{
							max_v = std::max(max_v, v[j][i]);
						}
					}
					for (int j = 0; j < omp_get_max_threads(); j++)
					{
						for (int i = 0; i < iterator.ws.part[j].second - iterator.ws.part[j].first; i++)
						{
							v[j][i] = v[j][i] - max_v;
						}
					}
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
							for (int j = 0; j < omp_get_max_threads(); j++)
							{
								for (int i = iterator.ws.part[j].first; i < iterator.ws.part[j].second; i++)
								{
									SC diff = v_list[l][i] - v[j][i - iterator.ws.part[j].first];
									dlt += diff;
									dlt2 += diff * diff;
								}
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

			// free reserved memory.
			for (int t = 0; t < omp_get_max_threads(); t++)
			{
				lapFree(colactive[t]);
				lapFree(d[t]);
				lapFree(v[t]);
			}
			lapFree(pred);
			lapFree(colactive);
			lapFree(d);
			lapFree(v);
			lapFree(colsol);
			lapFree(min_private);
			lapFree(jmin_private);
			lapFree(perm);
			lapFree(tt);
		}

		// shortcut for square problems
		template <class SC, class CF, class I>
		void solve(int dim, CF &costfunc, I &iterator, int *rowsol, bool use_epsilon)
		{
			lap::omp::solve<SC>(dim, dim, costfunc, iterator, rowsol, use_epsilon);
		}

		template <class SC, class CF>
		SC cost(int dim, int dim2, CF &costfunc, int *rowsol)
		{
			SC total = SC(0);
#pragma omp parallel for reduction(+:total)
			for (int i = 0; i < dim; i++) total += costfunc.getCost(i, rowsol[i]);
			return total;
		}

		template <class SC, class CF>
		SC cost(int dim, CF &costfunc, int *rowsol)
		{
			return lap::omp::cost<SC, CF>(dim, dim, costfunc, rowsol);
		}
	}
}
