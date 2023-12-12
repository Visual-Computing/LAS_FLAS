package de.htw.lcs.ag;

import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;
import it.unimi.dsi.fastutil.objects.ObjectArrayList;
import it.unimi.dsi.fastutil.objects.ObjectList;

public class Grid<T> {

	protected final int rows;
	protected final int columns;
	protected final T[] elements;
	
	protected final boolean[] fixedElements;
	
	public Grid(int columns, int rows, T[] elements) {
		this(columns, rows, Arrays.copyOf(elements, rows*columns), null);
	}
	
	public Grid(int columns, int rows, T[] elements, boolean[] fixedElements) {
		this.rows = rows;
		this.columns = columns;
		this.elements = elements;
		this.fixedElements = fixedElements;
	}

	public T[] getElements() {
		return elements;
	}

	public int getRows() {
		return rows;
	}

	public int getColumns() {
		return columns;
	}

	public T getElement(int x, int y) {
		if(x < 0 || x >= columns || y < 0 || y >= rows)
			return null;
		return elements[y*columns+x];		
	}
	
	public void setElement(int x, int y, T entry) {	
		if(x < 0 || x >= columns || y < 0 || y >= rows)
			return;
		
		elements[y*columns+x] = entry;
	}

	public T getElement(int index) {
		if(index < 0 || index >= elements.length)
			return null;
		return elements[index];
	}
	
	public void setElement(int index, T entry) {
		if(index < 0 || index >= elements.length)
			return;
		elements[index] = entry;
	}

	public int getSize() {
		return rows*columns;
	}

	/**
	 * Count how many map cells are not NULL
	 * 
	 * @return
	 */
	public int getElementCount() {
		int count = 0;
		for (int i = 0; i < elements.length; i++)
			if(elements[i] != null)
				count++;
		return count;
	}

	public boolean[] getFixedElements() {
		return fixedElements;
	}
	
	/**
	 * Shuffle all swappable elements in the grid
	 * 
	 * @param rnd
	 */
	public void shuffle(Random rnd) {
		
		// find swappable positions (not fixed and not a hole)
		final IntList swappablePos = new IntArrayList(elements.length);
		final ObjectList<T> swappableElement = new ObjectArrayList<>(elements.length);
		for (int i = 0; i < elements.length; i++) {
			if(elements[i] != null && (fixedElements != null && fixedElements[i] == false)) {
				swappablePos.add(i);
			}
		}
		
		// copy the position and shuffle the new list
		Collections.shuffle(swappablePos, rnd);
		
		// swap the positions of the swappable elements
		for (int i = 0; i < swappablePos.size(); i++) 
			elements[swappablePos.getInt(i)] = swappableElement.get(i);		
	}
}