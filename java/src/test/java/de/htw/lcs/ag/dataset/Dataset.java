package de.htw.lcs.ag.dataset;

import java.awt.image.BufferedImage;

import de.htw.lcs.ag.Describable;
import de.htw.lcs.ag.DistanceFunction;
import de.htw.lcs.ag.Grid;

public interface Dataset<T extends Describable> {
	
	/**
	 * Get the data set
	 * 
	 * @return
	 */
	public T[] getAll();
		
	/**
	 * Comparison function for different feature vectors of data set entries
	 * 
	 * @return 
	 */
	public DistanceFunction<float[]> getFeatureDistanceFunction();
	
	/**
	 * Visualize the current order of features vectors
	 * 
	 * @param featureMap
	 * @param mapWidth
	 * @param mapHeight
	 * @return
	 */
	public BufferedImage visualise(Grid<T> imageGrid);
}
