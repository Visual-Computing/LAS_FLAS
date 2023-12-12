package de.htw.lcs.ag.dataset;


import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;

import javax.imageio.ImageIO;

import de.htw.lcs.ag.Describable;
import de.htw.lcs.ag.DistanceFunction;
import de.htw.lcs.ag.Grid;

public class ColorDataset implements Dataset<ColorDataset.ColorData> {
	
	protected final ColorData[] colors;
	
	protected ColorDataset(ColorData[] colors) {
		this.colors = colors;	
		
	}

	@Override
	public ColorData[] getAll() {
		return colors.clone();
	}

	@Override
	public DistanceFunction<float[]> getFeatureDistanceFunction() {
		return (float[] element1, float[] element2) -> {
			return DistanceFunction.getSSEDistance(element1, element2);
		};
	}
	
	@Override
	public BufferedImage visualise(Grid<ColorData> imageGrid) {
		int mapWidth = imageGrid.getColumns();
		int mapHeight = imageGrid.getRows();
		
		// convert map places to pixels 
		int[] pixels = new int[mapWidth*mapHeight];
		for (int y = 0; y < mapHeight; y++) {
			for (int x = 0; x < mapWidth; x++) {
				int pos = y*mapWidth+x;	
				if (imageGrid.getElement(pos) != null) {
					int color = imageGrid.getElement(pos).getArgb();		
					pixels[pos] = color;
				}
				else
					pixels[pos] = 0xFF777777;
			}
		}
		
		BufferedImage image = new BufferedImage(mapWidth, mapHeight, BufferedImage.TYPE_INT_ARGB);
		image.setRGB(0, 0, mapWidth, mapHeight, pixels, 0, mapWidth);
		return image;
	}
		
	/**
	 * Single element of the color data set
	 */
	public static class ColorData implements Describable {
		
		protected final int index;
		protected final int argb;
		protected final float[] feature;
		
		public ColorData(int index, int argb, float[] feature) {
			this.index = index;
			this.argb = argb;
			this.feature = feature;
		}
		
		public int getArgb() {
			return argb;
		}
		
		@Override
		public float[] getFeature() {
			return feature;
		}
		
		@Override
		public int hashCode() {
			return index;
		}
		
		@Override
		public String toString() {
			return "Color"+index+"("+Arrays.toString(feature)+")";
		};
	}
	
	
	
	// --------------------------------------------------------------------------------------
	// ------------------------------ import functions --------------------------------------
	// --------------------------------------------------------------------------------------
	

	/**
	 * 
	 * @param steps
	 * @param imageSize
	 * @param path
	 * @throws IOException 
	 */
	public static ColorDataset loadDataset(Path imageFile) throws IOException {
		BufferedImage image = ImageIO.read(imageFile.toFile());
		
		int w = image.getWidth();
		int h = image.getHeight();
		int[] rgbArray = new int[w * h];
		image.getRGB(0, 0, w, h, rgbArray, 0, w);
		
		ColorData[] colors = new ColorData[rgbArray.length];
		for (int i = 0; i < rgbArray.length; i++) {
			int rgbValue = rgbArray[i];
			int red   = (rgbValue >> 16) & 255;
			int green = (rgbValue >>  8) & 255;
			int blue  = (rgbValue >>  0) & 255;
			int argb = 0xFF000000 | ((red & 0xFF) << 16) | ((green & 0xFF) << 8) | (blue & 0xFF); 
			colors[i] = new ColorData(i, argb, new float[] {red, green, blue});
		}
		System.out.println("Loaded "+colors.length+" colors from file "+imageFile);
		return new ColorDataset(colors);
	}
	
	/**
	 * 
	 * @param colorCount
	 * @param rnd
	 * @return
	 */
	public static ColorDataset createRandomColorDataset(int colorCount, Random rnd) {
		int mapSize = (int) Math.sqrt(colorCount);
		if(mapSize * mapSize != colorCount)
			throw new RuntimeException("Step size does not produce a good mapSize: steps^3 = mapSize^2");
		
		ColorData[] colors = new ColorData[colorCount];
		for (int i = 0; i < colorCount; i++) {
			int r = rnd.nextInt(256);
			int g = rnd.nextInt(256);
			int b = rnd.nextInt(256);
			int argb = 0xFF000000 | ((r & 0xFF) << 16) | ((g & 0xFF) << 8) | (b & 0xFF); 
			colors[i] = new ColorData(i, argb, new float[] { r, g, b });
		}
		
		return new ColorDataset(colors);
	}
	
	/**
	 * Create a color data set by sub sampling the RGB space with "steps" in each dimension
	 * 
	 * steps^3 = mapSize^2
	 * steps = 9 16 25 36 49
	 */
	public static ColorDataset createColorDataset(int steps) {
		int colorCount = steps * steps * steps;
		int mapSize = (int) Math.sqrt(colorCount);
		if(mapSize * mapSize != colorCount)
			throw new RuntimeException("Step size does not produce a good mapSize: steps^3 = mapSize^2");
		
		System.out.println("Create "+colorCount+" colors by subsamping the RGB space with a step size of "+steps);
		ColorData[] colors = new ColorData[steps*steps*steps];
		int stepSize = Math.max(1, 255 / steps);
		int stepSizeHalf = Math.max(0, (255 - (stepSize * steps - stepSize)) / 2);
				
		for (int rs = 0; rs < steps; rs++) {
			for (int gs = 0; gs < steps; gs++) {
				for (int bs = 0; bs < steps; bs++) {
					int r = stepSizeHalf + stepSize * rs;
					int g = stepSizeHalf + stepSize * gs;
					int b = stepSizeHalf + stepSize * bs;
					int argb = 0xFF000000 | ((r & 0xFF) << 16) | ((g & 0xFF) << 8) | (b & 0xFF); 
					int idx = rs*steps*steps + gs*steps + bs;
					colors[idx] = new ColorData(idx, argb, new float[] { r, g, b });
				}
			}
		}
		
		return new ColorDataset(colors);
	}
}