#pragma once
#include <vector>

namespace lap
{
	template <class T>
	class CacheListNode
	{
	public:
		T prev, next;
	};

	// segmented least recently used
	class CacheSLRU
	{
	protected:
		std::vector<CacheListNode<int>> list;
		int first[2];
		int last[2];
		std::vector<int> id;
		std::vector<char> priv;
		int priv_avail;
		long long chit, cmiss;
    std::vector<int> map;

		__forceinline void remove_entry(int i)
		{
			int l = priv[i];
			int prev = list[i].prev;
			int next = list[i].next;
			if (prev != -1) list[prev].next = next;
			if (next != -1) list[next].prev = prev;
			if (first[l] == i) first[l] = next;
			if (last[l] == i) last[l] = prev;
		}

		__forceinline void remove_first(int i)
		{
			int l = priv[i];
			first[l] = list[i].next;
			list[first[l]].prev = -1;
		}

		__forceinline void push_last(int i)
		{
			int l = priv[i];
			list[i].prev = last[l];
			list[i].next = -1;
			if (last[l] == -1)
			{
				first[l] = i;
			}
			else
			{
				list[last[l]].next = i;
			}
			last[l] = i;
		}

	public:
		CacheSLRU()
		{
			first[0] = first[1] = last[0] = last[1] = -1;
			priv_avail = 0;
			chit = cmiss = 0ll;
		}
		~CacheSLRU() {}

		__forceinline void setSize(int entries, int dim)
		{
      // calling map.resize() leads to a warning in gcc here so use this code instead
      {
        std::vector<int> tmp(dim);
        map.swap(tmp);
      }
			list.resize(entries);
			id.resize(entries);
			priv.resize(entries);
			for (int i = 0; i < dim; i++) map[i] = -1;
			for (int i = 0; i < entries; i++) { list[i].prev = i - 1; list[i].next = (i == entries - 1) ? -1 : i + 1; }
			for (int i = 0; i < entries; i++) id[i] = -1;
			for (int i = 0; i < entries; i++) priv[i] = 0;
			chit = cmiss = 0;
			first[0] = 0;
			last[0] = entries - 1;
			first[1] = -1;
			last[1] = -1;
			priv_avail = entries >> 1;
		}

		__forceinline bool find(int &idx, int i)
		{
			if (map[i] == -1)
			{
				// replace
				idx = first[0];
				if (id[idx] != -1) map[id[idx]] = -1;
				id[idx] = i;
				map[i] = idx;
				remove_first(idx);
				push_last(idx);
				cmiss++;
				return false;
			}
			else
			{
				idx = map[i];
				if (priv[idx] == -1)
				{
					priv[idx] = 0;
					remove_entry(idx);
				}
				else
				{
					remove_entry(idx);
					if (priv[idx] == 0)
					{
						priv[idx] = 1;
						if (priv_avail > 0)
						{
							priv_avail--;
						}
						else
						{
							int idx1 = first[1];
							remove_first(idx1);
							priv[idx1] = 0;
							push_last(idx1);
						}
					}
				}
				push_last(idx);
				chit++;
				return true;
			}
		}

		__forceinline void restart()
		{
			int entries = (int)list.size();
			for (int i = 0; i < entries; i++) { list[i].prev = i - 1; list[i].next = (i == entries - 1) ? -1 : i + 1; }
			for (int i = 0; i < entries; i++) priv[i] = -1;
			first[0] = 0;
			last[0] = entries - 1;
			first[1] = -1;
			last[1] = -1;
			priv_avail = entries >> 1;
		}

		__forceinline void getHitMiss(long long &hit, long long &miss) { hit = chit; miss = cmiss; chit = 0; cmiss = 0; }
		__forceinline int getEntries() { return (int)list.size(); }
	};

	// least frequently used
	class CacheLFU
	{
	protected:
		long long chit, cmiss;
		std::vector<int> map;
		std::vector<int> id;
		std::vector<int> count;
		std::vector<int> order;
		std::vector<int> pos;
		int entries;

		__forceinline void advance(int start)
		{
			// this uses a heap now
			bool done = false;
			int i = start;
			int ii = id[order[i]];
			int ci = count[ii];
			while (!done)
			{
				int l = i + i + 1;
				int r = l + 1;
				if (l >= entries) done = true;
				else
				{
					if (r >= entries)
					{
						int il = id[order[l]];
						int cl;
						if (il == -1) cl = -1; else cl = count[il];
						if ((ci > cl) || ((ci == cl) && (ii < il)))
						{
							std::swap(order[i], order[l]);
							pos[order[i]] = i;
							pos[order[l]] = l;
							i = l;
							ii = id[order[i]];
							ci = count[ii];
						}
						else
						{
							done = true;
						}
					}
					else
					{
						int il = id[order[l]];
						int ir = id[order[r]];
						int cl, cr;
						if (il == -1) cl = -1; else cl = count[il];
						if (ir == -1) cr = -1; else cr = count[ir];
						if ((cr > cl) || ((cr == cl) && (ir < il)))
						{
							// left
							if ((ci > cl) || ((ci == cl) && (ii < il)))
							{
								std::swap(order[i], order[l]);
								pos[order[i]] = i;
								pos[order[l]] = l;
								i = l;
							}
							else
							{
								done = true;
							}
						}
						else
						{
							// right
							if ((ci > cr) || ((ci == cr) && (ii < ir)))
							{
								std::swap(order[i], order[r]);
								pos[order[i]] = i;
								pos[order[r]] = r;
								i = r;
							}
							else
							{
								done = true;
							}
						}
					}
				}
			}
		}

	public:
		CacheLFU()
		{
			entries = 0;
			chit = cmiss = 0ll;
		}
		~CacheLFU() {}

		__forceinline void setSize(int p_entries, int dim)
		{
			entries = p_entries;
			map.resize(dim);
			count.resize(dim);
			id.resize(entries);
			order.resize(entries);
			pos.resize(entries);
			for (int i = 0; i < dim; i++) map[i] = -1;
			for (int i = 0; i < dim; i++) count[i] = 0;
			for (int i = 0; i < entries; i++) order[i] = i;
			for (int i = 0; i < entries; i++) pos[i] = i;
			for (int i = 0; i < entries; i++) id[i] = -1;
			chit = cmiss = 0;
		}

		__forceinline bool find(int &idx, int i)
		{
			if (map[i] == -1)
			{
				// replace
				idx = order[0];
				if (id[idx] != -1) map[id[idx]] = -1;
				id[idx] = i;
				map[i] = idx;
				count[i]++;
				advance(0);
				cmiss++;
				return false;
			}
			else
			{
				idx = map[i];
				count[i]++;
				advance(pos[idx]);
				chit++;
				return true;
			}
		}

		__forceinline void restart()
		{
			int dim = (int)count.size();
			for (int i = 0; i < dim; i++) count[i] = 0;
		}

		__forceinline void getHitMiss(long long &hit, long long &miss) { hit = chit; miss = cmiss; chit = 0; cmiss = 0; }
		__forceinline int getEntries() { return entries; }
	};
}
