package de.htw.lcs.ag;

import java.util.Random;

@FunctionalInterface
public interface ArrangingGrid<T> {	
    public Grid<T> doSorting(Grid<T> imageGrid);
    
    /**
     * Just shuffle the image grid
     * 
     * @param <T>
     * @param rndShuffle
     * @return
     */
    public static <T> ArrangingGrid<T> shuffle(Random rndShuffle) {
    	return new ArrangingGrid<T>() {

			@Override
			public Grid<T> doSorting(Grid<T> imageGrid) {
				imageGrid.shuffle(rndShuffle);
	    		return imageGrid;
			}
			
			@Override
			public String toString() {
				return "shuffle";
			}
		}; 
    }
    
    
    
    /**
     * Do not change anything in the grid
     * 
     * @param <T>
     * @return
     */
    public static <T> ArrangingGrid<T> identity() {
    	return new ArrangingGrid<T>() {

			@Override
			public Grid<T> doSorting(Grid<T> imageGrid) {
				return imageGrid;
			}
			
			@Override
			public String toString() {
				return "none";
			}
		}; 
    }
}
