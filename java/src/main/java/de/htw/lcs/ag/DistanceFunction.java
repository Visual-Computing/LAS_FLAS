package de.htw.lcs.ag;

@FunctionalInterface
public interface DistanceFunction<T> {
	
	/**
	 * compute the distance between the two elements
	 * 
	 * @param ele1
	 * @param ele2
	 * @return value between 0 and Float.MAX_VALUE
	 */
	public float calcDistance(T ele1, T ele2);
	
	
	/**
	 * http://www.math.uwaterloo.ca/tsp/world/geom.html
	 * 
	 * The feature vectors needs to contain the coords in longitude and latitude format
	 * 
	 * @param actFeatureData
	 * @param searchFeatureData
	 * @return
	 */
	public static float getGEOMDistance(final float[] actFeatureData, final float[] searchFeatureData) {
		double lati = actFeatureData[0], latj = searchFeatureData[0], longi = actFeatureData[1], longj = searchFeatureData[1];
		
		double q1 = Math.cos (latj) * Math.sin(longi - longj);
		double q3 = Math.sin((longi - longj)/2.0);
		double q4 = Math.cos((longi - longj)/2.0);
		double q2 = Math.sin(lati + latj) * q3 * q3 - Math.sin(lati - latj) * q4 * q4;
		double q5 = Math.cos(lati - latj) * q4 * q4 - Math.cos(lati + latj) * q3 * q3;
	    return (int) (6378388.0 * Math.atan2(Math.sqrt(q1*q1 + q2*q2), q5) + 1.0);
	}
	
	/**
	 * http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/TSPFAQ.html
	 * 
	 * The feature vectors needs to contain the coords in longitude and latitude format
	 * 
	 * @param actFeatureData
	 * @param searchFeatureData
	 * @return
	 */
	public static float getGEODistance(final float[] actFeatureData, final float[] searchFeatureData) {		
		double RRR = 6378.388;
		double q1 =  Math.cos( actFeatureData[0] - searchFeatureData[0] ); 	// longitude - longitude
		double q2 =  Math.cos( actFeatureData[1] - searchFeatureData[1] );	// latitude - latitude
		double q3 =  Math.cos( actFeatureData[1] + searchFeatureData[1] );	// latitude - latitude
		return (int) ( RRR * Math.acos( 0.5*((1.0+q1)*q2 - (1.0-q1)*q3) ) + 1.0);
	}
		
	
	public static float getSSEDistance(final float[] actFeatureData, final float[] searchFeatureData) {
		float dist = 0;
		for (int i = 0; i < actFeatureData.length; i++) {
			float d = actFeatureData[i] - searchFeatureData[i];
			dist += d*d;
		}
		return dist;
	}
	
	public static float getL2Distance(final float[] actFeatureData, final float[] searchFeatureData) {
		float dist = 0;
		for (int i = 0; i < actFeatureData.length; i++) {
			float d = actFeatureData[i] - searchFeatureData[i];
			dist += d*d;
		}
		return (float) Math.sqrt(dist);
	}
	
	public static float getL2pDistance(final float[] actFeatureData, final float[] searchFeatureData, final float L2_p) {
		float dist = 0;
		for (int i = 0; i < actFeatureData.length; i++) {
			float d = actFeatureData[i] - searchFeatureData[i];
			dist += d*d;
		}
		return (float) Math.pow(Math.sqrt(dist), L2_p);
	}
	
	
	public static float getL1Distance(final float[] actFeatureData, final float[] searchFeatureData) {
		float dist = 0;
		for (int i = 0; i < actFeatureData.length; i++) 
			dist += Math.abs(actFeatureData[i] - searchFeatureData[i]);
		return dist;
	}
}
