#include <iostream>
#include <fstream>
#include <sstream>
//openGPC library
#include "gpc/buffer.hpp"
//fadelib 
#include "fade/fade.hpp"
using namespace std;

int main(int argc, const char *argv[]) {
  std::string forestPath   = "../../forests/defaultZeroForest.txt";
  std::string leftImgPath = "../../data/kitti/img0l.png";
  std::string rightImgPath = "../../data/kitti/img0r.png";
  std::string outname;
  if(argc == 5){
    forestPath = argv[1];
    leftImgPath = argv[2];
    rightImgPath = argv[3];
    outname = argv[4];
  }else{
    cout << "Usage: " << argv[0] << " <forest path> <left image path> <right image path>" << endl;
    cout << "Trying defaults:" << endl;
    cout << "Forest path: " << forestPath << endl;
    cout << "Left image : " << leftImgPath << endl;
    cout << "Right image: " << leftImgPath << endl;
  }

  ndb::Fade fade;
  ndb::FadeSettings fadesettings = ndb::FadeSettings().builder()
      .niters(3)            // Use single refinement step
      .szz(32)              // initial refinement grid size: 32x32 pixels
      .tlo(0)               // Census cost lower bound
      .thi(3)               // Census cost upper bound
      .useFinalFilter(true) // Use final disparity filter based on lr-consistency cost
      .debug(false);        // No debug output

  gpc::inference::InferenceSettings gpcsettings = gpc::inference::InferenceSettings().builder()
    .gradientThreshold(7)
    .verticalTolerance(0)   // 0px tolerance for rectified epipolar matches
    .dispHigh(128)           // limit disparities to 64
    .epipolarMode(true)    // use global matching for GPC states. Higher accuracy than epipolar, but less matches.
    .useHashtable(false);   // use sort method for matching. faster for <100K descriptors.

  //Load image pair
  ndb::Buffer<uint8_t> simg, timg;
  if ( simg.readPNG(leftImgPath) || timg.readPNG(rightImgPath)) {
      cout << "No image data \n";
      return -1;
   }

  //define forest
  typedef gpc::inference::Forest GPCForest_t;

  GPCForest_t forest;
  GPCForest_t::FilterMask fm = forest.readForest(forestPath, simg.cols(), simg.rows()); 
 
  // Preprocess images (box filter, sobel filter, indices of high gradient pixels)
  gpc::inference::time_point t0 = gpc::inference::sysTick();
  GPCForest_t::PreprocessedImage simgP = forest.preprocessImage(simg, gpcsettings);
  GPCForest_t::PreprocessedImage timgP = forest.preprocessImage(timg, gpcsettings);
  gpc::inference::time_point t1 = gpc::inference::sysTick();
 
  ndb::DisparityResult disparityImage = fade.disparityEstimate(simgP, timgP, fm, gpcsettings, fadesettings);
  
  gpc::inference::time_point t2 = gpc::inference::sysTick();
  cout << "tPreprocess: " << gpc::inference::tickToMs(t1,t0) << " ms"
    << ", tMatch and iterate: " << gpc::inference::tickToMs(t2,t1)  << " ms"
    << ", num estimates:" << disparityImage.finalSupport.size() << endl;


  // Write disparity of support matches by openGPC (initial)
  // and the interpolated and filtered disparities (final)
  ndb::Buffer<ndb::RGBColor> renderDisp;
  renderDisp = ndb::getDisparityVisualization(simg, disparityImage.initialSupport);
  //renderDisp.writePNGRGB("disparity-fade-initial.png");
  renderDisp = ndb::getDisparityVisualization(simg, disparityImage.finalSupport);
  renderDisp.writePNGRGB(outname);
}
