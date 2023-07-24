package de.htw.lcs.ag;

import java.awt.BorderLayout;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import de.htw.lcs.ag.dataset.ColorDataset;
import de.htw.lcs.ag.dataset.Dataset;
import de.htw.lcs.ag.dataset.ImageDataset;
import de.htw.lcs.ag.evaluation.DistancePreservation;
import de.htw.lcs.ag.las.FLAS;
import de.htw.lcs.ag.las.LAS;

/**
 * The test will present the different layout constrains
 *
 * @author Nico Hezel
 *
 */
public class Examples {
	
	protected static enum ArrangeType { LAS, FLAS, SUFFLE, NONE }

	public static Path colorGridImage = Paths.get("test_images/colors_1024_random.png");
//	public static Path colorGridImage = Paths.get("test_images/rgb_4x4x4.png");
	public static Path imageDir = Paths.get("../data/web/");
	
	public static void main(String[] args) throws IOException {
		test_dense_map(ImageDataset.loadDataset(imageDir), ArrangeType.FLAS, false);
//		test_dense_map(ColorDataset.loadDataset(colorGridImage), ArrangeType.FLAS, false);
//		test_dense_map_1D(ColorDataset.loadDataset(colorGridImage), ArrangeType.FLAS, false);
//		test_holes_and_fixed_cells(ColorDataset.loadDataset(colorGridImage), ArrangeType.FLAS, false);
	}

	/**
	 * Arrange all the elements of the data set on a 2D grid
	 * 
	 * @param <T>
	 * @param dataset
	 * @param type
	 * @param doWrap
	 * @throws IOException
	 */
	public static <T extends Describable> void test_dense_map(Dataset<T> dataset, ArrangeType type, boolean doWrap) throws IOException {

		final T[] allElements = dataset.getAll();		
		final int[] mapSize = findMapSize(allElements.length);
		final int columns = mapSize[0];
		final int rows = mapSize[1];
		final Random rndShuffle = new Random(7);
		Collections.shuffle(Arrays.asList(allElements), rndShuffle);
		
		// fix the central element
		final Grid<T> imageGrid = new Grid<>(columns, rows, allElements);	
		
		displayArrangement(dataset, imageGrid, type, doWrap);		
	}
	
	/**
	 * Arrange all the elements of the data set on a 1D grid
	 * 
	 * @param <T>
	 * @param dataset
	 * @param type
	 * @param doWrap
	 * @throws IOException
	 */
	public static <T extends Describable> void test_dense_map_1D(Dataset<T> dataset, ArrangeType type, boolean doWrap) throws IOException {
		final T[] allElements = dataset.getAll();
		final int columns = allElements.length;
		final int rows = 1;
		final Random rndShuffle = new Random(7);
		Collections.shuffle(Arrays.asList(allElements), rndShuffle);
		
		// fix the central element
		final Grid<T> imageGrid = new Grid<>(columns, rows, allElements);	
		
		displayArrangement(dataset, imageGrid, type, doWrap);		
	}
	
	/**
	 * Double the size of the grid and freeze the center element in the grid.
	 * Then arrange all the elements of the data set on this grid.
	 * 
	 * @param <T>
	 * @param dataset
	 * @param type
	 * @param doWrap
	 * @throws IOException
	 */
	public static <T extends Describable> void test_holes_and_fixed_cells(Dataset<T> dataset, ArrangeType type, boolean doWrap) throws IOException {
		final T[] allElements = dataset.getAll();
		final int[] mapSize = findMapSize(allElements.length);
		int columns = mapSize[0];
		int rows = mapSize[1];
		
		// double the size of the grid
		columns = columns * 2;
		rows = rows * 1;
		
		// shuffle the data
		final Random rndShuffle = new Random(7);
		Collections.shuffle(Arrays.asList(allElements), rndShuffle);
		final T[] elements = Arrays.copyOf(allElements, rows*columns);
		
		// fix the central element
		final boolean[] fixedElements = new boolean[columns*rows];		
		fixedElements[(rows / 2) * columns + (columns / 2)] = true; // freeze cell in the center  
		final Grid<T> imageGrid = new Grid<>(columns, rows, elements, fixedElements);	
		
		displayArrangement(dataset, imageGrid, type, doWrap);		
	}
	
	
	/**
	 * Run the sorting algorithms and display the result
	 * 
	 * @param <T>
	 * @param dataset
	 * @param imageGrid
	 * @param type
	 * @param doWrap
	 * @throws IOException
	 */
	public static <T extends Describable> void displayArrangement(Dataset<T> dataset, Grid<T> imageGrid, ArrangeType type, boolean doWrap) throws IOException {
		final DistancePreservation<T> quality = new DistancePreservation<>(imageGrid.getElements(), 16);
		final Random rndShuffle = new Random(7);
		final Random rnd = new Random(7);
		
		if(imageGrid.getElementCount() != imageGrid.getSize() && ArrangeType.LAS == type)
		throw new RuntimeException("LAS does not support holes");
		
		if(imageGrid.getFixedElements() != null && ArrangeType.LAS == type)
			throw new RuntimeException("LAS does not support fixed elements");
		
		// sort the grid
		final ArrangingGrid<T> ag;		
		switch (type) {
			case FLAS:
				ag = new FLAS<>(dataset.getFeatureDistanceFunction(), rnd, doWrap);
				break;
			case LAS:
				ag = new LAS<>(dataset.getFeatureDistanceFunction(), rnd, doWrap);
				break;
			case SUFFLE:
				ag = ArrangingGrid.shuffle(rndShuffle);
				break;
			case NONE:
			default:
				ag = ArrangingGrid.identity();
				break;		
		}
		long start = System.currentTimeMillis();
		ag.doSorting(imageGrid);
		final long sortingTime = System.currentTimeMillis() - start;
		System.out.printf("Arranging the grid (%dx%d) with %s took %d ms\n", imageGrid.getColumns(), imageGrid.getRows(), ag.toString(), sortingTime);

		// check if every cell has a unique entry
		if(checkValidMap(imageGrid) == false) 
			System.out.println("Invalid map for "+ag.toString());

		// compute quality
		start = System.currentTimeMillis();			
		double sumQ = quality.computeQuality(imageGrid, doWrap);
		final long qualityTime = System.currentTimeMillis() - start;
		System.out.printf("Computing the quality %s (%.3f) took %d ms\n", quality.toString(), sumQ, qualityTime);

		// visualize the result
		final int scaleFactor = (dataset instanceof ColorDataset) ? 20 : 1;
		final BufferedImage bi = dataset.visualise(imageGrid);
		final Image image = bi.getScaledInstance(bi.getWidth()*scaleFactor, bi.getHeight()*scaleFactor, Image.SCALE_AREA_AVERAGING);

		final JFrame frame = new JFrame(ag.toString()+" sorting took "+sortingTime+"ms. DPQ_16="+String.format("%.3f", sumQ)+" took "+qualityTime+"ms");
		final JLabel label1=new JLabel();
		label1.setIcon(new ImageIcon(image));
		frame.getContentPane().add(label1, BorderLayout.CENTER);			
		frame.setLocationRelativeTo(null);
		frame.pack();
		frame.setVisible(true);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	}
	
	/**
	 * Find a width/height combination which has the minimum amount of holes and is close to quadratic.
	 * 
	 * @return [width, height]
	 */
	public static int[] findMapSize(int numOfElements) {
		
		int width=0, height=0;
		int mid = (int) Math.sqrt(numOfElements+1);
		
		int minDiff = Integer.MAX_VALUE; 
		double bestAngleDiff = Float.MAX_VALUE;
				
		for (int h = mid/2; h < 2*mid; h++) {
			for (int w = mid; w < 2*mid; w++) {
				int prod = w * h;
				int diff = prod - numOfElements;

				if (diff >= 0 && diff <= minDiff) {
					double angle = Math.atan2(h, w);
					double angleDiff = Math.abs(angle - Math.PI/4);

					if (angleDiff < 0.1*Math.PI/8) {
						minDiff = diff;

						if (angleDiff < bestAngleDiff) {
							bestAngleDiff = angleDiff;
							width = w;
							height = h;
						}
					}
				}
			}
		}
		
		return new int[] {width, height};
	}
	
	/**
	 * Check if each cell is occupied by a unique element
	 * 
	 * @param <T>
	 * @param elementMap
	 * @return
	 */
	protected static <T> boolean checkValidMap(Grid<T> imageGrid) {
		
		// count how often a index of an element is found in the map
		final Set<Integer> duplicates = new HashSet<>();
		final Set<Integer> setOfHashs = new HashSet<>();
		for (T element : imageGrid.getElements()) {
			if(element == null) continue;
			final int hash = element.hashCode();
			if(setOfHashs.add(hash) == false)
				duplicates.add(hash);
		}
		
		// list of all indices which occur multiple times in the map
		for (int hash : duplicates) {
			System.out.print("Hash "+hash+" occurs multiple times at positions: ");
			for (int y = 0; y < imageGrid.getRows(); y++) {
				for (int x = 0; x < imageGrid.getColumns(); x++) {
					final T element = imageGrid.getElement(x, y);
					if(element == null) continue;
					if(element.hashCode() == hash) 
						System.out.print("(x: "+x+",y: "+y+"),");
				}
			}
			System.out.println();
		}
		
		return duplicates.size() == 0;
	}
}