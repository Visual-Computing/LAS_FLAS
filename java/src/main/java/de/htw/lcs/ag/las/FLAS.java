package de.htw.lcs.ag.las;

import java.util.Arrays;
import java.util.Objects;
import java.util.Random;

import de.htw.lcs.ag.Describable;
import de.htw.lcs.ag.DistanceFunction;
import de.htw.lcs.ag.Grid;
import de.htw.lcs.ag.solver.JonkerVolgenantGo;

/*
 * Fast Linear Assignment Sorting
 * 
 * Can handle holes and fixed cells. 
 */
public class FLAS<T extends Describable> extends LAS<T> {
	
	protected static final float WeightHole = 0.01f;    	// TODO adjust to the amount of holes
	protected static final float WeightSwappable = 1f;  
	protected static final float WeightNonSwappable = 100f;
	
	protected static final int   QUANT = 256; // /256;  quantized distance steps
	
	// hyper parameter 
	public static int   MaxSwapPositions = 9;
	public static float SampleFactor = 1.0f;	// 1 the fraction of swaps per iteration
	static {
		FLAS.InitialRadiusFactor = 0.5f;
		FLAS.RadiusDecay = 0.93f;
		FLAS.EndRadius = 1.0f;
		FLAS.NumFilters = 1;
	}
		    
    // map variables
    private boolean[] fixedElements = null; // fixed map positions
    
    // temporary variables
    private int[] swapPositions;
    private T[] swapedElements;
    private float[][] fvs;
    private float[][] somFvs;

	public FLAS(DistanceFunction<float[]> distanceFunction, Random random, boolean doWrap) {
		super(distanceFunction, random, doWrap);
	}

	@Override
	public Grid<T> doSorting(Grid<T> imageGrid) {		
		this.fixedElements = imageGrid.getFixedElements();
		
		final int columns = imageGrid.getColumns();
		final int rows = imageGrid.getRows();		
		final int gridSize = columns*rows;
		final int dim = Arrays.stream(imageGrid.getElements()).filter(Objects::nonNull).mapToInt(e -> e.getFeature().length).max().orElse(-1);

		final float[][] som = new float[gridSize][dim];
		final float[] weights = new float[gridSize];
		
		// temporary variables using the maximal swap position count
		this.swapPositions = new int[Math.min(MaxSwapPositions, rows*columns)];
		this.swapedElements = Arrays.copyOf(imageGrid.getElements(), swapPositions.length);
		this.fvs = new float[swapPositions.length][];
		this.somFvs = new float[swapPositions.length][];
		this.distLut  = new int[swapPositions.length][swapPositions.length];
		this.distLutF  = new float[swapPositions.length][swapPositions.length];
		
		
		
		// setup the initial radius
		float rad = Math.max(columns, rows)*InitialRadiusFactor;	
			
		// try to improve the map
		do {
			final int radius = (int) Math.max(1, Math.round(rad));  // set the radius
			final int radiusX = Math.max(1, Math.min(columns/2, radius));
			final int radiusY = Math.max(1, Math.min(rows/2, radius));
			
			copyFeatureVectorsToSom(imageGrid, som, weights, dim);
			for (int i = 0; i < NumFilters; i++) 
				filterWeightedSom(radiusX, radiusY, columns, rows, som, dim, weights, doWrap);
			checkRandomSwaps(radius, imageGrid, som); 
			
			rad *= RadiusDecay;  
		}
		while (rad > EndRadius); 
		
		return imageGrid;
	}

	
	private void copyFeatureVectorsToSom(Grid<T> imageGrid, float[][] som, float[] weights, int dim) {
		final T[] elements = imageGrid.getElements();
		for (int pos = 0; pos < elements.length; pos++)  {
			
			final float[] somCell = som[pos];
					
			if (elements[pos] != null) {
				final float[] fv = elements[pos].getFeature();
				// higher weight for fixed images
				float w = fixedElements != null && fixedElements[pos] ? WeightNonSwappable : WeightSwappable; 
				for (int i = 0; i < fv.length; i++) 
					somCell[i] = w * fv[i];
				weights[pos] = w; 
			}
			else {
				for (int i = 0; i < dim; i++) 
					somCell[i] *= WeightHole;
				weights[pos] = WeightHole;
			}
		}
	}
	
	// -------------------------------------------------------------------------------------------------------------
	// ---------------------------------------- Filter part-------------------------------------------------
	// -------------------------------------------------------------------------------------------------------------
	
	protected static void filterWeightedSom(int actRadiusX, int actRadiusY, int columns, int rows, float[][] som, int dim, float[] weights, boolean doWrap) {

		int filterSizeX = 2*actRadiusX+1;
		int filterSizeY = 2*actRadiusY+1;

		float[][] somH = new float[rows * columns][dim];
		float[] weightsH = new float[rows * columns];
		
		if(doWrap) {
			filterHwrap(som, somH, rows, columns, dim, filterSizeX);
			filterHwrap(weights, weightsH, rows, columns, filterSizeX);
			
			filterVwrap(somH, som, rows, columns, dim, filterSizeY);	
			filterVwrap(weightsH, weights, rows, columns, filterSizeY);	
			
		}
		else {
			filterHmirror(som, somH, rows, columns, dim, filterSizeX);
			filterHmirror(weights, weightsH, rows, columns, filterSizeX);
			
			filterVmirror(somH, som, rows, columns, dim, filterSizeY);	
			filterVmirror(weightsH, weights, rows, columns, filterSizeY);	
		}	
		
		for (int i = 0; i < som.length; i++) {
			float w = 1 / weights[i];
			for (int d = 0; d < dim; d++) 
				som[i][d] *= w;		
		}
	}
	
	protected static void filterHwrap(float[] input, float[] output, int rows, int columns, int filterSize) {

		int ext = filterSize/2;							  // size of the border extension

		float[] rowExt = new float[columns + 2*ext];  // extended row

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

			float tmp = 0; 
			for (int i = 0; i < filterSize; i++) // first element
				tmp += rowExt[i];

			output[actRow] = tmp / filterSize;

			for (int i = 1; i < columns; i++) { // rest of the row
				int left = i-1;
				int right = left + filterSize;
				tmp += rowExt[right] - rowExt[left];
				output[actRow + i] = tmp / filterSize; 
			}
		}
	}
	
	protected static void filterVwrap(float[] input, float[] output, int rows, int columns, int filterSize) {
		
		int ext = filterSize/2;		// size of the border extension
		
		float[] colExt = new float[rows + 2*ext];  // extended row
		
		// filter the columns
		for (int x = 0; x < columns; x++) {

			for (int i = 0; i < rows; i++) 
				colExt[i+ext] = input[x + i*columns]; // copy one column 
		
			// wrapped extension
			for (int i = 0; i < ext; i++) {
				colExt[ext-1-i] = colExt[rows+ext-i-1];
				colExt[rows+ext+i] = colExt[ext+i];
			}

			float tmp = 0; 
			for (int i = 0; i < filterSize; i++) // first element
				tmp += colExt[i];

			output[x] = tmp / filterSize;

			for (int i = 1; i < rows; i++) { // rest of the column
				int left = i-1;
				int right = left + filterSize;
				
				tmp += colExt[right] - colExt[left];
				output[x + i*columns] = tmp / filterSize; 
			}
		}
	}
	
	protected static void filterHmirror(float[] input, float[] output, int rows, int columns, int filterSize) {

		int ext = filterSize/2;							  // size of the border extension

		float[] rowExt = new float[columns + 2*ext];  // extended row

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

			float tmp = 0; 
			for (int i = 0; i < filterSize; i++) // first element
				tmp += rowExt[i];

			output[actRow] = tmp / filterSize;

			for (int i = 1; i < columns; i++) { // rest of the row
				int left = i-1;
				int right = left + filterSize;

				tmp += rowExt[right] - rowExt[left];
				output[actRow + i] = tmp / filterSize; 
			}
		}
	}
	
	protected static void filterVmirror(float[] input, float[] output, int rows, int columns, int filterSize) {
		
		int ext = filterSize/2;		// size of the border extension
		
		float[] colExt = new float[rows + 2*ext];  // extended row
		
		// filter the columns
		for (int x = 0; x < columns; x++) {

			for (int i = 0; i < rows; i++) 
				colExt[i+ext] = input[x + i*columns]; // copy one column 

			// mirrored extension
			for (int i = 0; i < ext; i++) {
				colExt[ext-1-i] = colExt[ext+i+1];
				colExt[rows + ext + i] = colExt[ext+rows-2-i];
			}

			float tmp = 0; 
			for (int i = 0; i < filterSize; i++) // first element
				tmp += colExt[i];

			output[x] = tmp / filterSize;

			for (int i = 1; i < rows; i++) { // rest of the column
				int left = i-1;
				int right = left + filterSize;
				
				tmp += colExt[right] - colExt[left];
				output[x + i*columns] = tmp / filterSize; 
			}
		}
	}
	
	// -------------------------------------------------------------------------------------------------------------
	// ---------------------------------------- Swap and Solver part-------------------------------------------------
	// -------------------------------------------------------------------------------------------------------------
	
	private void checkRandomSwaps(int radius, Grid<T> imageGrid, float[][] som) {
		final int columns = imageGrid.getColumns();
		final int rows = imageGrid.getRows();
		
		// set swap size
		int swapAreaWidth = Math.min(2*radius+1, columns);
		int swapAreaHeight = Math.min(2*radius+1, rows);
		int k = 0;
		while (swapAreaHeight * swapAreaWidth < swapPositions.length) {
			if ((k++ & 0x1) == 0) // alternate the size increase
				swapAreaWidth = Math.min(swapAreaWidth+1, columns);
			else
				swapAreaHeight = Math.min(swapAreaHeight+1, rows);
		}	
				

		// get all positions of the actual swap region
		final int[] swapIndices = new int[swapAreaWidth*swapAreaHeight];
		for (int i = 0, y = 0; y < swapAreaHeight; y++)
			for (int x = 0; x < swapAreaWidth; x++)
				swapIndices[i++] = y*columns + x;
		shuffleArray(swapIndices, random);

		
		final int numSwapTries = (int) Math.max(1,(SampleFactor * rows * columns / swapPositions.length));
		if(doWrap) {
			for (int n = 0; n < numSwapTries; n++) {			
				final int numSwapPositions = findSwapPositionsWrap(swapIndices, swapPositions, swapAreaWidth, swapAreaHeight, rows, columns); 
				doSwaps(swapPositions, numSwapPositions, imageGrid, som);
			}	
		} else {
			for (int n = 0; n < numSwapTries; n++) {			
				final int numSwapPositions = findSwapPositions(swapIndices, swapPositions, swapAreaWidth, swapAreaHeight, rows, columns); 
				doSwaps(swapPositions, numSwapPositions, imageGrid, som);
			}	
		}
	}
	
	private static void shuffleArray(int[] array, Random random)
	{
		int index, temp;
		for (int i = array.length - 1; i > 0; i--)
		{
			index = random.nextInt(i + 1);
			temp = array[index];
			array[index] = array[i];
			array[i] = temp;
		}
	}
	
	
	private int findSwapPositionsWrap(int[] swapIndices, int[] swapPositions, int swapAreaWidth, int swapAreaHeight, int rows, int columns) {
		final int startIndex = (swapIndices.length - swapPositions.length > 0) ? random.nextInt(swapIndices.length - swapPositions.length) : 0;
		final int pos0 = random.nextInt(rows*columns);
		
		int numSwapPositions = 0;
		for (int j = startIndex; j < swapIndices.length && numSwapPositions < swapPositions.length; j++) {			
			int d = pos0 + swapIndices[j]; 
			int x = d % columns;
			int y = (d / columns) % rows;
			int pos = y * columns + x;

			if (fixedElements == null || fixedElements[pos] == false) 
				swapPositions[numSwapPositions++] = pos;
		}	
		
		return swapPositions.length;
	}


	private int findSwapPositions(int[] swapIndices, int[] swapPositions, int swapAreaWidth, int swapAreaHeight, int rows, int columns) {
		
		// calculate start position of swap area
		final int pos0 = random.nextInt(rows*columns);
		final int x0 =  pos0 % columns;
		final int y0 =  pos0 / columns;
		
		int xStart = Math.max(0, x0 - swapAreaWidth/2);     
		int yStart = Math.max(0, y0 - swapAreaHeight/2); 
		if (xStart + swapAreaWidth > columns)
			xStart = columns-swapAreaWidth;
		if (yStart + swapAreaHeight > rows)
			yStart = rows-swapAreaHeight;
		
		final int startIndex = (swapIndices.length - swapPositions.length > 0) ? random.nextInt(swapIndices.length - swapPositions.length) : 0;
		int numSwapPositions = 0;
		for (int j = startIndex; j < swapIndices.length && numSwapPositions < swapPositions.length; j++) {
			int dx = swapIndices[j] % columns;
			int dy = swapIndices[j] / columns;
			
			int x = (xStart + dx) % columns;
			int y = (yStart + dy) % rows;
			int pos = y * columns + x;
			
			if (fixedElements == null || fixedElements[pos] == false) 
				swapPositions[numSwapPositions++] = pos;
		}
		
		return numSwapPositions;
	}	

	private void doSwaps(int[] swapPositions, int numSwapPositions, Grid<T> imageGrid, float[][] som) { 
		
		int numValid = 0;
		for (int i = 0; i < numSwapPositions; i++) {
			final int swapPosition = swapPositions[i];
			final T swapedElement = swapedElements[i] = imageGrid.getElement(swapPosition);
			
			// handle holes
			if (swapedElement != null) {
				fvs[i] = swapedElement.getFeature();
				numValid++;
			}
			else 
				fvs[i] = som[swapPosition]; // hole
			
			somFvs[i] = som[swapPosition];
		}
					
		if (numValid > 0) {
			int[][] distLut = calcDistLutL2Int(fvs, somFvs, numSwapPositions);
			int[] permutation = JonkerVolgenantGo.computeAssignment(distLut, numSwapPositions);	

			for (int i = 0; i < numSwapPositions; i++) 
				imageGrid.setElement(swapPositions[permutation[i]], swapedElements[i]); 
		}
	}
	
	
	private int[][] calcDistLutL2Int(float[][] fv, float[][] mv, int size) {
		
		float max = 0;		
		for (int i = 0; i < size; i++) 
			for (int j = 0; j < size; j++) {
				float val = distLutF[i][j] = distFunc.calcDistance(fv[i], mv[j]);
				if (val > max)
					max = val;
			}
		
		for (int i = 0; i < size; i++) 
			for (int j = 0; j < size; j++) 
				distLut[i][j] = (int) (QUANT * distLutF[i][j] / max + 0.5);
		
		return distLut;
	}
	
	@Override
	public String toString() {
		return "FLAS";
	}
	
}