package de.htw.lcs.ag.las;

import java.util.Arrays;
import java.util.Objects;
import java.util.Random;

import de.htw.lcs.ag.ArrangingGrid;
import de.htw.lcs.ag.Describable;
import de.htw.lcs.ag.DistanceFunction;
import de.htw.lcs.ag.Grid;
import de.htw.lcs.ag.solver.JonkerVolgenantGo;

/*
 * Linear Assignment Sorting
 * 
 * Does not work for grids with holes. 
 */
public class LAS<T extends Describable> implements ArrangingGrid<T> {
	
	public static final int QUANT = 2048; 		// TODO  quantized distance
	public static final float SomAlpha = 0.5f;
		
	// hyper parameter 
	public static int NumFilters = 2;
	public static float InitialRadiusFactor = 0.5f; //0.35f;
	public static float RadiusDecay = 0.99f; // 0.9
	public static float EndRadius = 1.0f;
    
	protected final Random random;
	protected final DistanceFunction<float[]> distFunc;
	protected final boolean doWrap;
    
    protected float[][] distLutF;
    protected int[][] distLut;
    
	public LAS(DistanceFunction<float[]> distanceFunction, Random random, boolean doWrap) {
		this.random = random;
		this.distFunc = distanceFunction;
		this.doWrap = doWrap;
	}

	@Override
	public Grid<T> doSorting(Grid<T> imageGrid) {		
		final int columns = imageGrid.getColumns();
		final int rows = imageGrid.getRows();
		final int gridSize = columns*rows;
		final int dim = Arrays.stream(imageGrid.getElements()).filter(Objects::nonNull).mapToInt(e -> e.getFeature().length).max().orElse(-1);
		
		final float[][] som = new float[gridSize][dim];
		
		final T[] swapedElements = Arrays.copyOf(imageGrid.getElements(), gridSize);
		this.distLutF  = new float[gridSize][gridSize];
		this.distLut  = new int[gridSize][gridSize];
		
		// setup the initial radius
		float rad = Math.max(columns, rows)*InitialRadiusFactor; 	
		
		// try to improve the map
		do {				
			final int radius = (int) Math.max(1, rad);
			final int radiusX = Math.min(columns-1, radius);
			final int radiusY = Math.min(rows-1, radius);
			
			copyFeatureVectorsToSom(imageGrid, som, SomAlpha);
			for (int i = 0; i < NumFilters; i++) 
				filterSom(radiusX, radiusY, columns, rows, som, dim, doWrap);
			checkAllSwaps(imageGrid, som, swapedElements);
		
			rad =  rad*RadiusDecay;  
		}
		while (rad >= EndRadius); 

		return imageGrid;
	}


	private void copyFeatureVectorsToSom(Grid<T> imageGrid, float[][] som, float alpha) {
		final T[] elements = imageGrid.getElements();
		for (int pos = 0; pos < elements.length; pos++)  {
			final float[] fv = elements[pos].getFeature();
			final float[] somCell = som[pos];
			for (int i = 0; i < fv.length; i++) 
				somCell[i] = somCell[i] * alpha + fv[i] * (1-alpha);
		}
	}
		
	// -------------------------------------------------------------------------------------------------------------
	// ---------------------------------------- Filter part-------------------------------------------------
	// -------------------------------------------------------------------------------------------------------------
	
	protected static void filterSom(int actRadiusX, int actRadiusY, int columns, int rows, float[][] som, int dim, boolean doWrap) {

		int filterSizeX = 2*actRadiusX+1;
		int filterSizeY = 2*actRadiusY+1;

		float[][] somH = new float[rows * columns][dim];
		
		if(doWrap) {
			filterHwrap(som, somH, rows, columns, dim, filterSizeX);
			filterVwrap(somH, som, rows, columns, dim, filterSizeY);	
		}
		else {
			filterHmirror(som, somH, rows, columns, dim, filterSizeX);
			filterVmirror(somH, som, rows, columns, dim, filterSizeY);	
		}	
	}
	
	protected static void filterHwrap(float[][] input, float[][] output, int rows, int columns, int dims, int filterSize) {

		int ext = filterSize/2;							  // size of the border extension

		float[][] rowExt = new float[columns + 2*ext][];  // extended row

		// filter the rows
		for (int y = 0; y < rows; y++) {

			int actRow = y*columns;

			for (int i = 0; i < columns; i++) 
				rowExt[i+ext] = input[actRow + i]; // copy one row 

			// wrapped extension
			for (int i = 0; i < ext; i++) {
				rowExt[ext-1-i] = rowExt[columns+ext-i-1];
				rowExt[columns+ext+i] = rowExt[ext+i];
			}

			float[] tmp = new float[dims]; 
			for (int i = 0; i < filterSize; i++) // first element
				for (int d = 0; d < dims; d++) 
					tmp[d] += rowExt[i][d];

			for (int d = 0; d < dims; d++) 
				output[actRow][d] = tmp[d] / filterSize;

			for (int i = 1; i < columns; i++) { // rest of the row
				int left = i-1;
				int right = left + filterSize;

				for (int d = 0; d < dims; d++) { 
					tmp[d] += rowExt[right][d] - rowExt[left][d];
					output[actRow + i][d] = tmp[d] / filterSize; 
				}
			}
		}
	}
	
protected static void filterVwrap(float[][] input, float[][] output, int rows, int columns, int dims, int filterSize) {
		
		int ext = filterSize/2;		// size of the border extension
		
		float[][] colExt = new float[rows + 2*ext][];  // extended row
		
		// filter the columns
		for (int x = 0; x < columns; x++) {

			for (int i = 0; i < rows; i++) 
				colExt[i+ext] = input[x + i*columns]; // copy one column 
		
			// wrapped extension
			for (int i = 0; i < ext; i++) {
				colExt[ext-1-i] = colExt[rows+ext-i-1];
				colExt[rows+ext+i] = colExt[ext+i];
			}

			float[] tmp = new float[dims]; 
			for (int i = 0; i < filterSize; i++) // first element
				for (int d = 0; d < dims; d++) 
					tmp[d] += colExt[i][d];

			for (int d = 0; d < dims; d++) 
				output[x][d] = tmp[d] / filterSize;

			for (int i = 1; i < rows; i++) { // rest of the column
				int left = i-1;
				int right = left + filterSize;
				
				for (int d = 0; d < dims; d++) { 
					tmp[d] += colExt[right][d] - colExt[left][d];
					output[x + i*columns][d] = tmp[d] / filterSize; 
				}
			}
		}
	}

	protected static void filterHmirror(float[][] input, float[][] output, int rows, int columns, int dims, int filterSize) {
	
		int ext = filterSize/2;							  // size of the border extension
	
		float[][] rowExt = new float[columns + 2*ext][];  // extended row
	
		// filter the rows
		for (int y = 0; y < rows; y++) {
	
			int actRow = y*columns;
	
			for (int i = 0; i < columns; i++) 
				rowExt[i+ext] = input[actRow + i]; // copy one row 
	
			// mirrored extension
			for (int i = 0; i < ext; i++) {
				rowExt[ext-1-i] = rowExt[ext+i+1];
				rowExt[columns + ext+i] = rowExt[columns + ext -2 -i];
			}
	
			float[] tmp = new float[dims]; 
			for (int i = 0; i < filterSize; i++) // first element
				for (int d = 0; d < dims; d++) 
					tmp[d] += rowExt[i][d];
	
			for (int d = 0; d < dims; d++) 
				output[actRow][d] = tmp[d] / filterSize;
	
			for (int i = 1; i < columns; i++) { // rest of the row
				int left = i-1;
				int right = left + filterSize;
	
				for (int d = 0; d < dims; d++) { 
					tmp[d] += rowExt[right][d] - rowExt[left][d];
					output[actRow + i][d] = tmp[d] / filterSize; 
				}
			}
		}
	}
	
	protected static void filterVmirror(float[][] input, float[][] output, int rows, int columns, int dims, int filterSize) {
		
		int ext = filterSize/2;		// size of the border extension
		
		float[][] colExt = new float[rows + 2*ext][];  // extended row
		
		// filter the columns
		for (int x = 0; x < columns; x++) {

			for (int i = 0; i < rows; i++) 
				colExt[i+ext] = input[x + i*columns]; // copy one column 

			// mirrored extension
			for (int i = 0; i < ext; i++) {
				colExt[ext-1-i] = colExt[ext+i+1];
				colExt[rows + ext + i] = colExt[ext+rows-2-i];
			}

			float[] tmp = new float[dims]; 
			for (int i = 0; i < filterSize; i++) // first element
				for (int d = 0; d < dims; d++) 
					tmp[d] += colExt[i][d];

			for (int d = 0; d < dims; d++) 
				output[x][d] = tmp[d] / filterSize;

			for (int i = 1; i < rows; i++) { // rest of the column
				int left = i-1;
				int right = left + filterSize;
				
				for (int d = 0; d < dims; d++) { 
					tmp[d] += colExt[right][d] - colExt[left][d];
					output[x + i*columns][d] = tmp[d] / filterSize; 
				}
			}
		}
	}
	
	// -------------------------------------------------------------------------------------------------------------
	// ---------------------------------------- Solver part-------------------------------------------------
	// -------------------------------------------------------------------------------------------------------------
	
	private void checkAllSwaps(Grid<T> imageGrid, float[][] som, T[] swapedElements) {
		final int gridSize = imageGrid.getSize();
		
		// set up the array of feature vectors and som features
		final float[][] fvs = new float[gridSize][];		
		for (int i = 0; i < gridSize; i++) {
			swapedElements[i] = imageGrid.getElement(i);
			fvs[i] = swapedElements[i].getFeature();
		}
		
		distLut = calcDistLutL2Int(fvs, som);
		
		final int[] permutation = JonkerVolgenantGo.computeAssignment(distLut);		
		for (int i = 0; i < gridSize; i++) 
			imageGrid.setElement(permutation[i], swapedElements[i]); 
	}
	
	
	private int[][] calcDistLutL2Int(float[][] fv, float[][] mv) {
		float max = 0;
		
		for (int i = 0; i < fv.length; i++) 
			for (int j = 0; j < mv.length; j++) {
				float val = distLutF[i][j] = distFunc.calcDistance(fv[i], mv[j]);
				if (val > max)
					max = val;
			}
		
		for (int i = 0; i < fv.length; i++) 
			for (int j = 0; j < mv.length; j++) 
				distLut[i][j] = (int) (QUANT * distLutF[i][j] / max + 0.5) ;
		
		return distLut;
	}
	
	@Override
	public String toString() {
		return "LAS";
	}
}