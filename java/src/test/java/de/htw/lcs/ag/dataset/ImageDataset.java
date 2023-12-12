package de.htw.lcs.ag.dataset;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.ConvolveOp;
import java.awt.image.Kernel;
import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.StreamSupport;

import javax.imageio.ImageIO;

import de.htw.lcs.ag.Describable;
import de.htw.lcs.ag.DistanceFunction;
import de.htw.lcs.ag.Grid;

public class ImageDataset implements Dataset<ImageDataset.ImageData> {
	
	protected static int MinThumbnailSize = 32; 
	protected static Color backgroundColor = Color.GRAY;
	protected static float borderRatio = 0; // 0.025f;
	protected static boolean useLowpass = false;
	
	
	protected final ImageData[] images;
		
	public ImageDataset(ImageData[] images) {
		this.images = images;
	}

	@Override
	public ImageData[] getAll() {
		return images.clone();
	}
	
	@Override
	public DistanceFunction<float[]> getFeatureDistanceFunction() {
		return (float[] element1, float[] element2) -> {
			return DistanceFunction.getSSEDistance(element1, element2);
		};
	}

	@Override
	public BufferedImage visualise(Grid<ImageData> imageGrid) {
		final int mapWidth = imageGrid.getColumns();
		final int mapHeight = imageGrid.getRows();
		
		final int thumbSize = MinThumbnailSize;
		final int imageWidth = mapWidth * thumbSize;
		final int imageHeight = mapHeight * thumbSize;
		BufferedImage gridImage = new BufferedImage(imageWidth, imageHeight, BufferedImage.TYPE_INT_ARGB);		
	    Graphics2D g = gridImage.createGraphics();	
	    g.setColor(backgroundColor);							//TODO
	    g.fillRect(0, 0, imageWidth, imageHeight);
	    
	    int kernelSize = 1;	    
	    float data[] =  new float[kernelSize*kernelSize];
	    for (int i = 0; i < data.length; i++) 
			data[i] = 1f / (kernelSize*kernelSize);
	    
	    Kernel kernel = new Kernel(kernelSize, kernelSize, data);
	    ConvolveOp convolve = new ConvolveOp(kernel, ConvolveOp.EDGE_NO_OP, null);

	    int bOffset = Math.round(borderRatio*thumbSize);
		for (int y = 0; y < mapHeight; y++) {
			for (int x = 0; x < mapWidth; x++) {
				if(imageGrid.getElement(x, y) == null)
					continue;
				
				BufferedImage bi = imageGrid.getElement(x, y).getImage();
				BufferedImage biDest = new BufferedImage(bi.getWidth(), bi.getHeight(), BufferedImage.TYPE_INT_ARGB);
			    
			    if (useLowpass) 
					convolve.filter(bi, biDest);
				else
					biDest = bi;
				
				int xOffset = x * thumbSize;
				int yOffset = y * thumbSize;
				g.drawImage(biDest, xOffset+bOffset, yOffset+bOffset, xOffset+thumbSize-bOffset, yOffset+thumbSize-bOffset, 0, 0, bi.getWidth(), bi.getHeight(), null);
			}
		}
		
		return gridImage;
	}
	
	
	/**
	 * Single element of the image data set
	 */
	public static class ImageData implements Describable {	
	
		protected final int index;
		protected final BufferedImage image;
		protected final float[] feature;
		protected final String filename;
		
		/**
		 * Sorting a feature vector for an image (which might be stored on disc)
		 * 
		 * @param index
		 * @param image
		 * @param feature
		 * @param category optional
		 * @param filename optional
		 */
		public ImageData(int index, BufferedImage image, float[] feature, String filename) {
			this.index = index;
			this.image = image;
			this.feature = feature;
			this.filename = filename;
		}
		
		@Override
		public float[] getFeature() {
			return feature;
		}
		
		@Override
		public int hashCode() {
			return index;
		}
		
		public String getFilename() {
			return filename;
		}
		
		public BufferedImage getImage() {
			return image;
		}
		
		@Override
		public String toString() {
			return "Image"+index+"("+filename+")";
		};
	}
	
	

	// --------------------------------------------------------------------------------------
	// ------------------------------ import functions --------------------------------------
	// --------------------------------------------------------------------------------------
	
	public static ImageDataset loadDataset(Path imageDir) throws IOException {	
		return loadDataset(imageDir, 4);
	}
	
	public static ImageDataset loadDataset(Path imageDir, int meanColorThumbSize) throws IOException {	
		final List<ImageData> images = new ArrayList<>();
		
		final Path vectorDir = imageDir.resolve("vectors");
		final boolean useVectorDir = Files.exists(vectorDir);
		
		try(DirectoryStream<Path> fileStream = Files.newDirectoryStream(imageDir, "*.{jpg,png}")) {
			StreamSupport.stream(fileStream.spliterator(), false)
		    .sorted(Comparator.comparing(Path::toString))
		    .forEach(imageFile -> {  
		    		 try {
						final BufferedImage image = ImageIO.read(imageFile.toFile());
						final String filename = imageFile.getFileName().toString();				
						final float[] feature = useVectorDir ? loadPaperVectors(vectorDir, filename) : extractMeanColor(image, meanColorThumbSize);
						images.add(new ImageData(images.size(), image, feature, imageFile.getFileName().toString()));
					} catch (IOException e) {
						e.printStackTrace();
					}
		    });	
			
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		System.out.println("Load "+images.size()+" images  from file "+imageDir);
		return new ImageDataset(images.toArray(new ImageData[images.size()]));
	}
	
	/**
	 * Load the vectors from the numpy files of the paper
	 * 
	 * @param vectorDir
	 * @param filename
	 * @return
	 * @throws IOException 
	 */
	protected static float[] loadPaperVectors(Path vectorDir, String filename) throws IOException {
		filename = filename.substring(0, filename.lastIndexOf(".")) + '.' + "npy";
		final byte[] b =  Files.readAllBytes(vectorDir.resolve(filename));
		final float[] fv = new float[50];
		for (int i = 0; i < fv.length; i++) 
			fv[i] = b[b.length-50 + i];		 
		return fv;
	}
	
	/**
	 * Compute the mean RGB color of the image
	 * 
	 * @param image
	 * @return
	 */
	protected static float[] extractMeanColor(BufferedImage bi, int thumbnailSize) {
		final Image scaledImage = bi.getScaledInstance(thumbnailSize, thumbnailSize, Image.SCALE_AREA_AVERAGING);
		
		// Convert Image to BufferedImage
        BufferedImage image = toBufferedImage(scaledImage);

        // read the pixels
		int width  = image.getWidth();
		int height = image.getHeight();
		int[] rgbArray = new int[width*height]; 
		image.getRGB(0, 0, width, height, rgbArray, 0, width); 
		
		// convert to float array of all pixels and their RGB channels
		float[] feature = new float[rgbArray.length*3]; 
		for (int i = 0; i < rgbArray.length; i++) {
			int rgb = rgbArray[i];
			feature[i*3+0] = (rgb>>16)&255;
			feature[i*3+1] = (rgb>> 8)&255;
			feature[i*3+2] = (rgb    )&255;
		}

		return feature;
	}
	
	/**
	 * Helper method to convert Image to BufferedImage
	 * 
	 * @param image
	 * @return
	 */
    private static BufferedImage toBufferedImage(Image image) {
        if (image instanceof BufferedImage) {
            return (BufferedImage) image;
        }

        BufferedImage bufferedImage = new BufferedImage(
            image.getWidth(null),
            image.getHeight(null),
            BufferedImage.TYPE_INT_ARGB
        );

        Graphics g = bufferedImage.getGraphics();
        g.drawImage(image, 0, 0, null);
        g.dispose();

        return bufferedImage;
    }
}
