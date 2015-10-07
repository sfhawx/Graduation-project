// Copyright (c) 2008-2011, Guoshen Yu <yu@cmap.polytechnique.fr>
// Copyright (c) 2008-2011, Jean-Michel Morel <morel@cmla.ens-cachan.fr>
//
// WARNING: 
// This file implements an algorithm possibly linked to the patent
//
// Jean-Michel Morel and Guoshen Yu, Method and device for the invariant 
// affine recognition recognition of shapes (WO/2009/150361), patent pending. 
//
// This file is made available for the exclusive aim of serving as
// scientific tool to verify of the soundness and
// completeness of the algorithm description. Compilation,
// execution and redistribution of this file may violate exclusive
// patents rights in certain countries.
// The situation being different for every country and changing
// over time, it is your responsibility to determine which patent
// rights restrictions apply to you before you compile, use,
// modify, or redistribute this file. A patent lawyer is qualified
// to make this determination.
// If and only if they don't conflict with any patent terms, you
// can benefit from the following license terms attached to this
// file.
//
// This program is provided for scientific and educational only:
// you can use and/or modify it for these purposes, but you are
// not allowed to redistribute this work or derivative works in
// source or executable form. A license must be obtained from the
// patent right holders for any other use.
//
// 
//*----------------------------- demo_ASIFT  --------------------------------*/
// Detect corresponding points in two images with the ASIFT method. 

// Please report bugs and/or send comments to Guoshen Yu yu@cmap.polytechnique.fr
// 
// Reference: J.M. Morel and G.Yu, ASIFT: A New Framework for Fully Affine Invariant Image 
//            Comparison, SIAM Journal on Imaging Sciences, vol. 2, issue 2, pp. 438-469, 2009. 
// Reference: ASIFT online demo (You can try ASIFT with your own images online.) 
//			  http://www.ipol.im/pub/algo/my_affine_sift/
/*---------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector>

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

using namespace std;

#ifdef _OPENMP
#include <omp.h>
#endif

#include "demo_lib_sift.h"
#include "io_png.h"

#include "library.h"
#include "frot.h"
#include "fproj.h"
#include "compute_asift_keypoints.h"
#include "compute_asift_matches.h"


# define IM_X 800
# define IM_Y 600

/**************************** input arguments ***************************/
/*********************************Absolute Tilt Tests/painting zoom x10***************************/
//char *input_img1 = "../experiment images/Absolute Tilt Tests/painting zoom x10/adam_zoom10_front.png";
//char *input_img2 = "../experiment images/Absolute Tilt Tests/painting zoom x10/adam_zoom10_45deg.png";
//char *input_img1 = "../experiment images/Absolute Tilt Tests/painting zoom x10/adam_zoom10_front.png";
//char *input_img2 = "../experiment images/Absolute Tilt Tests/painting zoom x10/adam_zoom10_45degR.png";
//char *input_img1 = "../experiment images/Absolute Tilt Tests/painting zoom x10/adam_zoom10_front.png";
//char *input_img2 = "../experiment images/Absolute Tilt Tests/painting zoom x10/adam_zoom10_65deg.png";
//char *input_img1 = "../experiment images/Absolute Tilt Tests/painting zoom x10/adam_zoom10_front.png";
//char *input_img2 = "../experiment images/Absolute Tilt Tests/painting zoom x10/adam_zoom10_65degR.png";
//char *input_img1 = "../experiment images/Absolute Tilt Tests/painting zoom x10/adam_zoom10_front.png";
//char *input_img2 = "../experiment images/Absolute Tilt Tests/painting zoom x10/adam_zoom10_75deg.png";
//char *input_img1 = "../experiment images/Absolute Tilt Tests/painting zoom x10/adam_zoom10_front.png";
//char *input_img2 = "../experiment images/Absolute Tilt Tests/painting zoom x10/adam_zoom10_75degR.png";
//char *input_img1 = "../experiment images/Absolute Tilt Tests/painting zoom x10/adam_zoom10_front.png";
//char *input_img2 = "../experiment images/Absolute Tilt Tests/painting zoom x10/adam_zoom10_80deg.png";
//char *input_img1 = "../experiment images/Absolute Tilt Tests/painting zoom x10/adam_zoom10_front.png";
//char *input_img2 = "../experiment images/Absolute Tilt Tests/painting zoom x10/adam_zoom10_80degR.png";
//char *output_img1 = "../resultImages/ver_zoom_x10_45.png";
//char *output_img2 = "../resultImages/hor_zoom_x10_45.png";
//char *output_img1 = "../resultImages/ver_zoom_x10_45R.png";
//char *output_img2 = "../resultImages/hor_zoom_x10_45R.png";
//char *output_img1 = "../resultImages/ver_zoom_x10_65.png";
//char *output_img2 = "../resultImages/hor_zoom_x10_65.png";
//char *output_img1 = "../resultImages/ver_zoom_x10_65R.png";
//char *output_img2 = "../resultImages/hor_zoom_x10_65R.png";
//char *output_img1 = "../resultImages/ver_zoom_x10_75.png";
//char *output_img2 = "../resultImages/hor_zoom_x10_75.png";
//char *output_img1 = "../resultImages/ver_zoom_x10_75R.png";
//char *output_img2 = "../resultImages/hor_zoom_x10_75R.png";
//char *output_img1 = "../resultImages/ver_zoom_x10_80.png";
//char *output_img2 = "../resultImages/hor_zoom_x10_80.png";
//char *output_img1 = "../resultImages/ver_zoom_x10_80R.png";
//char *output_img2 = "../resultImages/hor_zoom_x10_80R.png";
/**************************** magazine zoom x4 ***************************/


//char *input_img1 = "../experiment images/Absolute Tilt Tests/magazine zoom x4/V1_front.png";
//char *input_img2 = "../experiment images/Absolute Tilt Tests/magazine zoom x4/V1_10deg.png";
//char *input_img1 = "../experiment images/Absolute Tilt Tests/magazine zoom x4/V1_front.png";
//char *input_img2 = "../experiment images/Absolute Tilt Tests/magazine zoom x4/V1_20deg.png";
//char *input_img1 = "../experiment images/Absolute Tilt Tests/magazine zoom x4/V1_front.png";
//char *input_img2 = "../experiment images/Absolute Tilt Tests/magazine zoom x4/V1_30deg.png";
//char *input_img1 = "../experiment images/Absolute Tilt Tests/magazine zoom x4/V1_front.png";
//char *input_img2 = "../experiment images/Absolute Tilt Tests/magazine zoom x4/V1_40deg.png";
//char *input_img1 = "../experiment images/Absolute Tilt Tests/magazine zoom x4/V1_front.png";
//char *input_img2 = "../experiment images/Absolute Tilt Tests/magazine zoom x4/V1_50deg.png";
//char *input_img1 = "../experiment images/Absolute Tilt Tests/magazine zoom x4/V1_front.png";
//char *input_img2 = "../experiment images/Absolute Tilt Tests/magazine zoom x4/V1_60deg.png";
//char *input_img1 = "../experiment images/Absolute Tilt Tests/magazine zoom x4/V1_front.png";
//char *input_img2 = "../experiment images/Absolute Tilt Tests/magazine zoom x4/V1_70deg.png";
//char *input_img1 = "../experiment images/Absolute Tilt Tests/magazine zoom x4/V1_front.png";
//char *input_img2 = "../experiment images/Absolute Tilt Tests/magazine zoom x4/V1_80deg.png";
//char *output_img1 = "../resultImages/ver_magazine zoom x4_10.png";
//char *output_img2 = "../resultImages/hor_magazine zoom x4_10.png";
//char *output_img1 = "../resultImages/ver_magazine zoom x4_20.png";
//char *output_img2 = "../resultImages/hor_magazine zoom x4_20.png";
//char *output_img1 = "../resultImages/ver_magazine zoom x4_30.png";
//char *output_img2 = "../resultImages/hor_magazine zoom x4_30.png";
//char *output_img1 = "../resultImages/ver_magazine zoom x4_40.png";
//char *output_img2 = "../resultImages/hor_magazine zoom x4_40.png";
//char *output_img1 = "../resultImages/ver_magazine zoom x4_50.png";
//char *output_img2 = "../resultImages/hor_magazine zoom x4_50.png";
//char *output_img1 = "../resultImages/ver_magazine zoom x4_60.png";
//char *output_img2 = "../resultImages/hor_magazine zoom x4_60.png";
//char *output_img1 = "../resultImages/ver_magazine zoom x4_70.png";
//char *output_img2 = "../resultImages/hor_magazine zoom x4_70.png";
//char *output_img1 = "../resultImages/ver_magazine zoom x4_80.png";
//char *output_img2 = "../resultImages/hor_magazine zoom x4_80.png";


/**************************** painting zoom x1 ***************************/


//char *input_img1 = "../experiment images/Absolute Tilt Tests/painting zoom x1/adam_zoom1_front.png";
//char *input_img2 = "../experiment images/Absolute Tilt Tests/painting zoom x1/adam_zoom1_45deg.png";
//char *input_img1 = "../experiment images/Absolute Tilt Tests/painting zoom x1/adam_zoom1_front.png";
//char *input_img2 = "../experiment images/Absolute Tilt Tests/painting zoom x1/adam_zoom1_45degR.png";
//char *input_img1 = "../experiment images/Absolute Tilt Tests/painting zoom x1/adam_zoom1_front.png";
//char *input_img2 = "../experiment images/Absolute Tilt Tests/painting zoom x1/adam_zoom1_65deg.png";
//char *input_img1 = "../experiment images/Absolute Tilt Tests/painting zoom x1/adam_zoom1_front.png";
//char *input_img2 = "../experiment images/Absolute Tilt Tests/painting zoom x1/adam_zoom1_65degR.png";
//char *input_img1 = "../experiment images/Absolute Tilt Tests/painting zoom x1/adam_zoom1_front.png";
//char *input_img2 = "../experiment images/Absolute Tilt Tests/painting zoom x1/adam_zoom1_75deg.png";
//char *input_img1 = "../experiment images/Absolute Tilt Tests/painting zoom x1/adam_zoom1_front.png";
//char *input_img2 = "../experiment images/Absolute Tilt Tests/painting zoom x1/adam_zoom1_75degR.png";
//char *input_img1 = "../experiment images/Absolute Tilt Tests/painting zoom x1/adam_zoom1_front.png";
//char *input_img2 = "../experiment images/Absolute Tilt Tests/painting zoom x1/adam_zoom1_80deg.png";
//char *input_img1 = "../experiment images/Absolute Tilt Tests/painting zoom x1/adam_zoom1_front.png";
//char *input_img2 = "../experiment images/Absolute Tilt Tests/painting zoom x1/adam_zoom1_80degR.png";
//char *output_img1 = "../resultImages/ver_zoom_x1_45.png";
//char *output_img2 = "../resultImages/hor_zoom_x1_45.png";
//char *output_img1 = "../resultImages/ver_zoom_x1_45R.png";
//char *output_img2 = "../resultImages/hor_zoom_x1_45R.png";
//char *output_img1 = "../resultImages/ver_zoom_x1_65.png";
//char *output_img2 = "../resultImages/hor_zoom_x1_65.png";
//char *output_img1 = "../resultImages/ver_zoom_x1_65R.png";
//char *output_img2 = "../resultImages/hor_zoom_x1_65R.png";
//char *output_img1 = "../resultImages/ver_zoom_x1_75.png";
//char *output_img2 = "../resultImages/hor_zoom_x1_75.png";
//char *output_img1 = "../resultImages/ver_zoom_x1_75R.png";
//char *output_img2 = "../resultImages/hor_zoom_x1_75R.png";
//char *output_img1 = "../resultImages/ver_zoom_x1_80.png";
//char *output_img2 = "../resultImages/hor_zoom_x1_80.png";
//char *output_img1 = "../resultImages/ver_zoom_x1_80R.png";
//char *output_img2 = "../resultImages/hor_zoom_x1_80R.png";


/***********************************Mikolajczyk************************/
/***************************bark***************************/
//char *input_img1 = "../experiment images/Mikolajczyk/bark/img1.png";
//char *input_img2 = "../experiment images/Mikolajczyk/bark/img2.png";
//char *input_img2 = "../experiment images/Mikolajczyk/bark/img3.png";
//char *input_img2 = "../experiment images/Mikolajczyk/bark/img4.png";
//char *input_img2 = "../experiment images/Mikolajczyk/bark/img5.png";
//char *input_img2 = "../experiment images/Mikolajczyk/bark/img6.png";

//char *output_img1 = "../resultImages/ver_bark12.png";
//char *output_img2 = "../resultImages/hor_bark12.png";
//char *output_img1 = "../resultImages/ver_bark13.png";
//char *output_img2 = "../resultImages/hor_bark13.png";
//char *output_img1 = "../resultImages/ver_bark14.png";
//char *output_img2 = "../resultImages/hor_bark14.png";
//char *output_img1 = "../resultImages/ver_bark15.png";
//char *output_img2 = "../resultImages/hor_bark15.png";
//char *output_img1 = "../resultImages/ver_bark16.png";
//char *output_img2 = "../resultImages/hor_bark16.png";

/***************************bikes***************************/
//char *input_img1 = "../experiment images/Mikolajczyk/bikes/img1.png";
//char *input_img2 = "../experiment images/Mikolajczyk/bikes/img2.png";
//char *input_img2 = "../experiment images/Mikolajczyk/bikes/img3.png";
//char *input_img2 = "../experiment images/Mikolajczyk/bikes/img4.png";
//char *input_img2 = "../experiment images/Mikolajczyk/bikes/img5.png";
//char *input_img2 = "../experiment images/Mikolajczyk/bikes/img6.png";
//
//char *output_img1 = "../resultImages/ver_bikes12.png";
//char *output_img2 = "../resultImages/hor_bikes12.png";
//char *output_img1 = "../resultImages/ver_bikes13.png";
//char *output_img2 = "../resultImages/hor_bikes13.png";
//char *output_img1 = "../resultImages/ver_bikes14.png";
//char *output_img2 = "../resultImages/hor_bikes14.png";
//char *output_img1 = "../resultImages/ver_bikes15.png";
//char *output_img2 = "../resultImages/hor_bikes15.png";
//char *output_img1 = "../resultImages/ver_bikes16.png";
//char *output_img2 = "../resultImages/hor_bikes16.png";

/***************************boat***************************/
//char *input_img1 = "../experiment images/Mikolajczyk/boat/img1.png";
//char *input_img2 = "../experiment images/Mikolajczyk/boat/img2.png";
//char *input_img2 = "../experiment images/Mikolajczyk/boat/img3.png";
//char *input_img2 = "../experiment images/Mikolajczyk/boat/img4.png";
//char *input_img2 = "../experiment images/Mikolajczyk/boat/img5.png";
//char *input_img2 = "../experiment images/Mikolajczyk/boat/img6.png";

//char *output_img1 = "../resultImages/ver_boat12.png";
//char *output_img2 = "../resultImages/hor_boat12.png";
//char *output_img1 = "../resultImages/ver_boat13.png";
//char *output_img2 = "../resultImages/hor_boat13.png";
//char *output_img1 = "../resultImages/ver_boat14.png";
//char *output_img2 = "../resultImages/hor_boat14.png";
//char *output_img1 = "../resultImages/ver_boat15.png";
//char *output_img2 = "../resultImages/hor_boat15.png";
//char *output_img1 = "../resultImages/ver_boat16.png";
//char *output_img2 = "../resultImages/hor_boat16.png";


/***************************graf***************************/
//char *input_img1 = "../experiment images/Mikolajczyk/graf/img1.png";
//char *input_img2 = "../experiment images/Mikolajczyk/graf/img2.png";
//char *input_img2 = "../experiment images/Mikolajczyk/graf/img3.png";
//char *input_img2 = "../experiment images/Mikolajczyk/graf/img4.png";
//char *input_img2 = "../experiment images/Mikolajczyk/graf/img5.png";
//char *input_img2 = "../experiment images/Mikolajczyk/graf/img6.png";

//char *output_img1 = "../resultImages/ver_graf12.png";
//char *output_img2 = "../resultImages/hor_graf12.png";
//char *output_img1 = "../resultImages/ver_graf13.png";
//char *output_img2 = "../resultImages/hor_graf13.png";
//char *output_img1 = "../resultImages/ver_graf14.png";
//char *output_img2 = "../resultImages/hor_graf14.png";
//char *output_img1 = "../resultImages/ver_graf15.png";
//char *output_img2 = "../resultImages/hor_graf15.png";
//char *output_img1 = "../resultImages/ver_graf16.png";
//char *output_img2 = "../resultImages/hor_graf16.png";

/***************************leuven***************************/
//char *input_img1 = "../experiment images/Mikolajczyk/leuven/img1.png";
//char *input_img2 = "../experiment images/Mikolajczyk/leuven/img2.png";
//char *input_img2 = "../experiment images/Mikolajczyk/leuven/img3.png";
//char *input_img2 = "../experiment images/Mikolajczyk/leuven/img4.png";
//char *input_img2 = "../experiment images/Mikolajczyk/leuven/img5.png";
//char *input_img2 = "../experiment images/Mikolajczyk/leuven/img6.png";

//char *output_img1 = "../resultImages/ver_leuven12.png";
//char *output_img2 = "../resultImages/hor_leuven12.png";
//char *output_img1 = "../resultImages/ver_leuven13.png";
//char *output_img2 = "../resultImages/hor_leuven13.png";
//char *output_img1 = "../resultImages/ver_leuven14.png";
//char *output_img2 = "../resultImages/hor_leuven14.png";
//char *output_img1 = "../resultImages/ver_leuven15.png";
//char *output_img2 = "../resultImages/hor_leuven15.png";
//char *output_img1 = "../resultImages/ver_leuven16.png";
//char *output_img2 = "../resultImages/hor_leuven16.png";


/***************************trees***************************/
//char *input_img1 = "../experiment images/Mikolajczyk/trees/img1.png";
//char *input_img2 = "../experiment images/Mikolajczyk/trees/img2.png";
//char *input_img2 = "../experiment images/Mikolajczyk/trees/img3.png";
//char *input_img2 = "../experiment images/Mikolajczyk/trees/img4.png";
//char *input_img2 = "../experiment images/Mikolajczyk/trees/img5.png";
//char *input_img2 = "../experiment images/Mikolajczyk/trees/img6.png";

//char *output_img1 = "../resultImages/ver_trees12.png";
//char *output_img2 = "../resultImages/hor_trees12.png";
//char *output_img1 = "../resultImages/ver_trees13.png";
//char *output_img2 = "../resultImages/hor_trees13.png";
//char *output_img1 = "../resultImages/ver_trees14.png";
//char *output_img2 = "../resultImages/hor_trees14.png";
//char *output_img1 = "../resultImages/ver_trees15.png";
//char *output_img2 = "../resultImages/hor_trees15.png";
//char *output_img1 = "../resultImages/ver_trees16.png";
//char *output_img2 = "../resultImages/hor_trees16.png";

/***************************ubc***************************/
//char *input_img1 = "../experiment images/Mikolajczyk/ubc/img1.png";
//char *input_img2 = "../experiment images/Mikolajczyk/ubc/img2.png";
//char *input_img2 = "../experiment images/Mikolajczyk/ubc/img3.png";
//char *input_img2 = "../experiment images/Mikolajczyk/ubc/img4.png";
//char *input_img2 = "../experiment images/Mikolajczyk/ubc/img5.png";
//char *input_img2 = "../experiment images/Mikolajczyk/ubc/img6.png";

//char *output_img1 = "../resultImages/ver_ubc12.png";
//char *output_img2 = "../resultImages/hor_ubc12.png";
//char *output_img1 = "../resultImages/ver_ubc13.png";
//char *output_img2 = "../resultImages/hor_ubc13.png";
//char *output_img1 = "../resultImages/ver_ubc14.png";
//char *output_img2 = "../resultImages/hor_ubc14.png";
//char *output_img1 = "../resultImages/ver_ubc15.png";
//char *output_img2 = "../resultImages/hor_ubc15.png";
//char *output_img1 = "../resultImages/ver_ubc16.png";
//char *output_img2 = "../resultImages/hor_ubc16.png";

/***************************wall***************************/
//char *input_img1 = "../experiment images/Mikolajczyk/wall/img1.png";
//char *input_img2 = "../experiment images/Mikolajczyk/wall/img2.png";
//char *input_img2 = "../experiment images/Mikolajczyk/wall/img3.png";
//char *input_img2 = "../experiment images/Mikolajczyk/wall/img4.png";
//char *input_img2 = "../experiment images/Mikolajczyk/wall/img5.png";
//char *input_img2 = "../experiment images/Mikolajczyk/wall/img6.png";

//char *output_img1 = "../resultImages/ver_wall12.png";
//char *output_img2 = "../resultImages/hor_wall12.png";
//char *output_img1 = "../resultImages/ver_wall13.png";
//char *output_img2 = "../resultImages/hor_wall13.png";
//char *output_img1 = "../resultImages/ver_wall14.png";
//char *output_img2 = "../resultImages/hor_wall14.png";
//char *output_img1 = "../resultImages/ver_wall15.png";
//char *output_img2 = "../resultImages/hor_wall15.png";
//char *output_img1 = "../resultImages/ver_wall16.png";
//char *output_img2 = "../resultImages/hor_wall16.png";

/*******************************************************************************/
/******************************************************************************************/
char *input_img1 = "../experiment images/Transition Tilt Tests/t2/t2_0deg.png";
//char *input_img2 = "../experiment images/Transition Tilt Tests/t2/t2_10deg.png";
//char *input_img2 = "../experiment images/Transition Tilt Tests/t2/t2_20deg.png";
//char *input_img2 = "../experiment images/Transition Tilt Tests/t2/t2_30deg.png";
//char *input_img2 = "../experiment images/Transition Tilt Tests/t2/t2_40deg.png";
//char *input_img2 = "../experiment images/Transition Tilt Tests/t2/t2_50deg.png";
//char *input_img2 = "../experiment images/Transition Tilt Tests/t2/t2_60deg.png";
char *input_img2 = "../experiment images/Transition Tilt Tests/t2/t2_70deg.png";
//char *input_img2 = "../experiment images/Transition Tilt Tests/t2/t2_80deg.png";
//char *input_img2 = "../experiment images/Transition Tilt Tests/t2/t2_90deg.png";

//char *output_img1 = "../resultImages/ver_t2_10.png";
//char *output_img2 = "../resultImages/hor_t2_10.png";
//char *output_img1 = "../resultImages/ver_t2_20.png";
//char *output_img2 = "../resultImages/hor_t2_20.png";
//char *output_img1 = "../resultImages/ver_t2_30.png";
//char *output_img2 = "../resultImages/hor_t2_30.png";
//char *output_img1 = "../resultImages/ver_t2_40.png";
//char *output_img2 = "../resultImages/hor_t2_40.png";
//char *output_img1 = "../resultImages/ver_t2_50.png";
//char *output_img2 = "../resultImages/hor_t2_50.png";
//char *output_img1 = "../resultImages/ver_t2_60.png";
//char *output_img2 = "../resultImages/hor_t2_60.png";
char *output_img1 = "../resultImages/ver_t2_70.png";
char *output_img2 = "../resultImages/hor_t2_70.png";
//char *output_img1 = "../resultImages/ver_t2_80.png";
//char *output_img2 = "../resultImages/hor_t2_80.png";
//char *output_img1 = "../resultImages/ver_t2_90.png";
//char *output_img2 = "../resultImages/hor_t2_90.png";

/**********************************************************************************/

//char *input_img1 = "../experiment images/Transition Tilt Tests/t4/t4_0deg.png";
//char *input_img2 = "../experiment images/Transition Tilt Tests/t4/t4_10deg.png";
//char *input_img2 = "../experiment images/Transition Tilt Tests/t4/t4_20deg.png";
//char *input_img2 = "../experiment images/Transition Tilt Tests/t4/t4_30deg.png";
//char *input_img2 = "../experiment images/Transition Tilt Tests/t4/t4_40deg.png";
//char *input_img2 = "../experiment images/Transition Tilt Tests/t4/t4_50deg.png";
//char *input_img2 = "../experiment images/Transition Tilt Tests/t4/t4_60deg.png";
//char *input_img2 = "../experiment images/Transition Tilt Tests/t4/t4_70deg.png";
//char *input_img2 = "../experiment images/Transition Tilt Tests/t4/t4_80deg.png";
//char *input_img2 = "../experiment images/Transition Tilt Tests/t4/t4_90deg.png";

//char *output_img1 = "../resultImages/ver_t4_10.png";
//char *output_img2 = "../resultImages/hor_t4_10.png";
//char *output_img1 = "../resultImages/ver_t4_20.png";
//char *output_img2 = "../resultImages/hor_t4_20.png";
//char *output_img1 = "../resultImages/ver_t4_30.png";
//char *output_img2 = "../resultImages/hor_t4_30.png";
//char *output_img1 = "../resultImages/ver_t4_40.png";
//char *output_img2 = "../resultImages/hor_t4_40.png";
//char *output_img1 = "../resultImages/ver_t4_50.png";
//char *output_img2 = "../resultImages/hor_t4_50.png";
//char *output_img1 = "../resultImages/ver_t4_60.png";
//char *output_img2 = "../resultImages/hor_t4_60.png";
//char *output_img1 = "../resultImages/ver_t4_70.png";
//char *output_img2 = "../resultImages/hor_t4_70.png";
//char *output_img1 = "../resultImages/ver_t4_80.png";
//char *output_img2 = "../resultImages/hor_t4_80.png";
//char *output_img1 = "../resultImages/ver_t4_90.png";
//char *output_img2 = "../resultImages/hor_t4_90.png";
char *feat1 = "../feat1.txt";
char *feat2 = "../feat2.txt";
char *match_feat = "../match.txt";
char *opt_select = "0";

// int main(int argc, char **argv)
int main(void)
{
	IplImage* img1, *img2;
	std::cerr << " ******************************************************************************* " << std::endl
		<< " ***************************  ASIFT image matching  **************************** " << std::endl
		<< " ******************************************************************************* " << std::endl
		<< " imgIn1.png imgIn2.png imgOutVert.png imgOutHori.png " << std::endl
		<< "           matchings.txt keys1.txt keys2.txt [Resize option: 0/1] " << std::endl
		<< "- imgIn1.png, imgIn2.png: input images (in PNG format). " << std::endl
		<< "- imgOutVert.png, imgOutHori.png: output images (vertical/horizontal concatenated, " << std::endl
		<< "  in PNG format.) The detected matchings are connected by write lines." << std::endl
		<< "- matchings.txt: coordinates of matched points (col1, row1, col2, row2). " << std::endl
		<< "- keys1.txt keys2.txt: ASIFT keypoints of the two images." << std::endl
		<< "- [optional 0/1]. 1: input images resize to 800x600 (default). 0: no resize. " << std::endl 
		<< " ******************************************************************************* " << std::endl
		<< " *********************  X.Y.Sun, 2015.10.4 ******************** " << std::endl
		<< " ******************************************************************************* " << std::endl;
	// Read image1
	//读取图片1存入iarr1中
	float *iarr1;
	size_t w1, h1;
	if (NULL == (iarr1 = read_png_f32_gray(input_img1, &w1, &h1))) {
		std::cerr << "Unable to load image file " << input_img1 << std::endl;
		return 1;
	}
	//用数组初始化vector对象
	std::vector<float> ipixels1(iarr1, iarr1 + w1 * h1);
	free(iarr1); /*memcheck*/

	// Read image2
	//读取图片2存入iarr1中
	float * iarr2;
	size_t w2, h2;
	if (NULL == (iarr2 = read_png_f32_gray(input_img2, &w2, &h2))) {
		std::cerr << "Unable to load image file " << input_img2 << std::endl;
		return 1;
	}
	std::vector<float> ipixels2(iarr2, iarr2 + w2 * h2);
	free(iarr2); /*memcheck*/	

	//Resize the images to area wS*hW in remaining the apsect-ratio	
	//Resize if the resize flag is not set or if the flag is set unequal to 0

	float wS = IM_X;
	float hS = IM_Y;

	float zoom1=0, zoom2=0;	
	int wS1=0, hS1=0, wS2=0, hS2=0;
	vector<float> ipixels1_zoom, ipixels2_zoom;	

	int flag_resize = 0;
	if ( strcmp(opt_select, "0") == 0 )
	{	
		flag_resize = atoi(opt_select);
	}

	//if ((argc == 8) || (flag_resize != 0))
	if(flag_resize != 0)
	{
		cout << "WARNING: The input images are resized to " << wS << "x" << hS << " for ASIFT. " << endl 
			<< "         But the results will be normalized to the original image size." << endl << endl;

		float InitSigma_aa = 1.6;

		float fproj_p, fproj_bg;
		char fproj_i;
		float *fproj_x4, *fproj_y4;
		int fproj_o;

		fproj_o = 3;
		fproj_p = 0;
		fproj_i = 0;
		fproj_bg = 0;
		fproj_x4 = 0;
		fproj_y4 = 0;
		// calculate the area of the resized image
		float areaS = wS * hS;

		// Resize image 1 
		float area1 = w1 * h1;
		zoom1 = sqrt(area1/areaS);//计算出源图片和放缩图片的面积比的平方根

		wS1 = (int) (w1 / zoom1);//after resize image, the image width
		hS1 = (int) (h1 / zoom1);//the image height

		int fproj_sx = wS1;
		int fproj_sy = hS1;     

		float fproj_x1 = 0;
		float fproj_y1 = 0;
		float fproj_x2 = wS1;
		float fproj_y2 = 0;
		float fproj_x3 = 0;	     
		float fproj_y3 = hS1;

		/* Anti-aliasing filtering along vertical direction */
		/*对原图像进行模糊处理，通过X方向和Y方向的高斯模糊, 去除噪声，反走样*/
		if ( zoom1 > 1 )
		{
			float sigma_aa = InitSigma_aa * zoom1 / 2;
			//对源图像进行Y方向的高斯模糊
			GaussianBlur1D(ipixels1,w1,h1,sigma_aa,1);
			//对源图像进行X方向的高斯模糊
			GaussianBlur1D(ipixels1,w1,h1,sigma_aa,0);
		}

		// simulate a tilt: subsample the image along the vertical axis by a factor of t.
		ipixels1_zoom.resize(wS1*hS1);
		//在绝对斜度为t的条件下， 在Y方向进行采样
		/*
		ipixels1 经过模糊处理后的原始图像
		ipixels1_zoom 600 * 800 的空图像
		w1, h1 原始图像的宽和高
		fproj_sx, fproj_sy 变换以后图像的宽和高
		fproj_bg 背景色
		fproj_o   样条插值时候采用几次样条插值
		fproj_p 
		fproj_x1, fproj_y1 图像的左上角
		fproj_x2, fproj_y2 图像的右上角
		fproj_x3, fproj_y3 图像的左下角
		fproj_x4, fproj_y4 作者本意是做透视变换的，但是最后没做， 这个参数没用

		*/
		fproj (ipixels1, ipixels1_zoom, w1, h1, &fproj_sx, &fproj_sy, &fproj_bg, &fproj_o, &fproj_p, 
			&fproj_i , fproj_x1 , fproj_y1 , fproj_x2 , fproj_y2 , fproj_x3 , fproj_y3, fproj_x4, fproj_y4); 


		// Resize image 2 
		float area2 = w2 * h2;
		zoom2 = sqrt(area2/areaS);

		wS2 = (int) (w2 / zoom2);
		hS2 = (int) (h2 / zoom2);

		fproj_sx = wS2;
		fproj_sy = hS2;     

		fproj_x2 = wS2;
		fproj_y3 = hS2;

		/* Anti-aliasing filtering along vertical direction */
		if ( zoom1 > 1 )
		{
			float sigma_aa = InitSigma_aa * zoom2 / 2;
			GaussianBlur1D(ipixels2,w2,h2,sigma_aa,1);
			GaussianBlur1D(ipixels2,w2,h2,sigma_aa,0);
		}

		// simulate a tilt: subsample the image along the vertical axis by a factor of t.
		ipixels2_zoom.resize(wS2*hS2);
		fproj (ipixels2, ipixels2_zoom, w2, h2, &fproj_sx, &fproj_sy, &fproj_bg, &fproj_o, &fproj_p, 
			&fproj_i , fproj_x1 , fproj_y1 , fproj_x2 , fproj_y2 , fproj_x3 , fproj_y3, fproj_x4, fproj_y4); 
	}
	else 
	{
		printf("没有进行放缩进行匹配！");
		ipixels1_zoom.resize(w1*h1);	
		ipixels1_zoom = ipixels1;
		wS1 = w1;
		hS1 = h1;
		zoom1 = 1;

		ipixels2_zoom.resize(w2*h2);	
		ipixels2_zoom = ipixels2;
		wS2 = w2;
		hS2 = h2;
		zoom2 = 1;
	}


	// Compute ASIFT keypoints
	// number N of tilts to simulate t = 1, \sqrt{2}, (\sqrt{2})^2, ..., {\sqrt{2}}^(N-1)
	int num_of_tilts1 = 7;
	int num_of_tilts2 = 7;
	//	int num_of_tilts1 = 1;
	//	int num_of_tilts2 = 1;
	int verb = 0;
	// Define the SIFT parameters
	// 定义sift 的各种参数
	siftPar siftparameters;	
	//初始化sift的各种参数
	default_sift_parameters(siftparameters);
	//存取各张图片的特征点
	vector< vector< keypointslist > > keys1;		
	vector< vector< keypointslist > > keys2;	

	int num_keys1=0, num_keys2=0;


	cout << "Computing keypoints on the two images..." << endl;
	time_t tstart, tend;	
	tstart = time(0);
	//计算模仿的两个图的描述子
	num_keys1 = compute_asift_keypoints(ipixels1_zoom, wS1, hS1, num_of_tilts1, verb, keys1, siftparameters);
	num_keys2 = compute_asift_keypoints(ipixels2_zoom, wS2, hS2, num_of_tilts2, verb, keys2, siftparameters);

	tend = time(0);
	cout << "Keypoints computation accomplished in " << difftime(tend, tstart) << " seconds." << endl;

	//// Match ASIFT keypoints
	int num_matchings;
	matchingslist matchings;	
	cout << "Matching the keypoints..." << endl;
	tstart = time(0);
	num_matchings = compute_asift_matches(num_of_tilts1, num_of_tilts2, wS1, hS1, wS2, 
		hS2, verb, keys1, keys2, matchings, siftparameters);
	tend = time(0);
	cout << "Keypoints matching accomplished in " << difftime(tend, tstart) << " seconds." << endl;

	///////////////// Output image containing line matches (the two images are concatenated one above the other)
	int band_w = 20; // insert a black band of width band_w between the two images for better visibility

	int wo =  MAX(w1,w2);
	int ho = h1+h2+band_w;

	float *opixelsASIFT = new float[wo*ho];

	for(int j = 0; j < (int) ho; j++)
		for(int i = 0; i < (int) wo; i++)  opixelsASIFT[j*wo+i] = 255;		

	/////////////////////////////////////////////////////////////////// Copy both images to output
	for(int j = 0; j < (int) h1; j++)
		for(int i = 0; i < (int) w1; i++)  opixelsASIFT[j*wo+i] = ipixels1[j*w1+i];				

	for(int j = 0; j < (int) h2; j++)
		for(int i = 0; i < (int) (int)w2; i++)  opixelsASIFT[(h1 + band_w + j)*wo + i] = ipixels2[j*w2 + i];	

	//////////////////////////////////////////////////////////////////// Draw matches
	matchingslist::iterator ptr = matchings.begin();
	for(int i=0; i < (int) matchings.size(); i++, ptr++)
	{		
		draw_line(opixelsASIFT, (int) (zoom1*ptr->first.x), (int) (zoom1*ptr->first.y), 
			(int) (zoom2*ptr->second.x), (int) (zoom2*ptr->second.y) + h1 + band_w, 255.0f, wo, ho);		
	}

	///////////////////////////////////////////////////////////////// Save imgOut	
	write_png_f32(output_img1, opixelsASIFT, wo, ho, 1);

	/****测试代码****/
	
	img1 = cvLoadImage( output_img1, 1);
	cvNamedWindow("IMG_MATCH1", 1);//创建窗口  
	cvShowImage("IMG_MATCH1",img1);//显示
	cvWaitKey( 0 );
	
	/****************/

	delete[] opixelsASIFT; /*memcheck*/

	/////////// Output image containing line matches (the two images are concatenated one aside the other)
	int woH =  w1+w2+band_w;
	int hoH = MAX(h1,h2);

	float *opixelsASIFT_H = new float[woH*hoH];

	for(int j = 0; j < (int) hoH; j++)
		for(int i = 0; i < (int) woH; i++)  opixelsASIFT_H[j*woH+i] = 255;

	/////////////////////////////////////////////////////////////////// Copy both images to output
	for(int j = 0; j < (int) h1; j++)
		for(int i = 0; i < (int) w1; i++)  opixelsASIFT_H[j*woH+i] = ipixels1[j*w1+i];				

	for(int j = 0; j < (int) h2; j++)
		for(int i = 0; i < (int) w2; i++)  opixelsASIFT_H[j*woH + w1 + band_w + i] = ipixels2[j*w2 + i];	


	//////////////////////////////////////////////////////////////////// Draw matches
	matchingslist::iterator ptrH = matchings.begin();
	for(int i=0; i < (int) matchings.size(); i++, ptrH++)
	{		
		draw_line(opixelsASIFT_H, (int) (zoom1*ptrH->first.x), (int) (zoom1*ptrH->first.y), 
			(int) (zoom2*ptrH->second.x) + w1 + band_w, (int) (zoom2*ptrH->second.y), 255.0f, woH, hoH);		
	}

	///////////////////////////////////////////////////////////////// Save imgOut	
	write_png_f32(output_img2, opixelsASIFT_H, woH, hoH, 1);

	/****测试代码.X.Y.Sun 2014****/
	img2 = cvLoadImage( output_img2, 1);
	cvNamedWindow("IMG_MATCH2", 1);//创建窗口  
	cvShowImage("IMG_MATCH2",img2);//显示
	cvWaitKey( 0 );
	/****************/

	delete[] opixelsASIFT_H; /*memcheck*/

	////// Write the coordinates of the matched points (row1, col1, row2, col2) to the file match_feat
	std::ofstream file(match_feat);
	if (file.is_open())
	{		
		// Write the number of matchings in the first line
		file << num_matchings << std::endl;

		matchingslist::iterator ptr = matchings.begin();
		for(int i=0; i < (int) matchings.size(); i++, ptr++)		
		{
			file << zoom1*ptr->first.x << "  " << zoom1*ptr->first.y << "  " <<  zoom2*ptr->second.x << 
				"  " <<  zoom2*ptr->second.y << std::endl;
		}		
	}
	else 
	{
		std::cerr << "Unable to open the file matchings."; 
	}

	file.close();



	// Write all the keypoints (row, col, scale, orientation, desciptor (128 integers)) to 
	// the file feat1 (so that the users can match the keypoints with their own matching algorithm if they wish to)
	// keypoints in the 1st image
	std::ofstream file_key1(feat1);
	if (file_key1.is_open())
	{
		// Follow the same convention of David Lowe: 
		// the first line contains the number of keypoints and the length of the desciptors (128)
		file_key1 << num_keys1 << "  " << VecLength << "  " << std::endl;
		for (int tt = 0; tt < (int) keys1.size(); tt++)
		{
			for (int rr = 0; rr < (int) keys1[tt].size(); rr++)
			{
				keypointslist::iterator ptr = keys1[tt][rr].begin();
				for(int i=0; i < (int) keys1[tt][rr].size(); i++, ptr++)	
				{
					file_key1 << zoom1*ptr->x << "  " << zoom1*ptr->y << "  " << zoom1*ptr->scale << "  " << ptr->angle;

					for (int ii = 0; ii < (int) VecLength; ii++)
					{
						file_key1 << "  " << ptr->vec[ii];
					}

					file_key1 << std::endl;
				}
			}	
		}
	}
	else 
	{
		std::cerr << "Unable to open the file keys1."; 
	}

	file_key1.close();

	// keypoints in the 2nd image
	std::ofstream file_key2(feat2);
	if (file_key2.is_open())
	{
		// Follow the same convention of David Lowe: 
		// the first line contains the number of keypoints and the length of the desciptors (128)
		file_key2 << num_keys2 << "  " << VecLength << "  " << std::endl;
		for (int tt = 0; tt < (int) keys2.size(); tt++)
		{
			for (int rr = 0; rr < (int) keys2[tt].size(); rr++)
			{
				keypointslist::iterator ptr = keys2[tt][rr].begin();
				for(int i=0; i < (int) keys2[tt][rr].size(); i++, ptr++)	
				{
					file_key2 << zoom2*ptr->x << "  " << zoom2*ptr->y << "  " << zoom2*ptr->scale << "  " << ptr->angle;

					for (int ii = 0; ii < (int) VecLength; ii++)
					{
						file_key2 << "  " << ptr->vec[ii];
					}					
					file_key2 << std::endl;
				}
			}	
		}
	}
	else 
	{
		std::cerr << "Unable to open the file keys2."; 
	}
	file_key2.close();

	return 0;
}
