// Copyright (c) 2018, ETH Zurich
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Implements the method proposed in
// The High-Performance and Tunable Stereo Reconstruction
// Sudeep Pillai Srikumar Ramalingam, John J Leonard
// ICRA 2016
// Code Author: Niklaus Bamert (bamertn@ethz.ch)
#ifndef _NDB_FastAdaptiveDepthEstimation
#define _NDB_FastAdaptiveDepthEstimation

#include <iostream>

//GPC includes
#include "gpc/inference.hpp"
//The triangle library
#include "fade/triangle/triangle.h"
// Templated 2D- buffer type with various operations
#include "gpc/buffer.hpp"

#include <nmmintrin.h>//For popcnt method
#include <thread>
#include <algorithm>
#include <chrono>
#include <bitset>
#include <cassert>
#include <set>

using namespace std;


namespace ndb {
struct FadeSettings{
  // Number of iterations to run
  int niters_ = 2;
  // initial regular rectangular grid dimensions
  // for finding refinement points. szz is halved
  // in each iteration
  int szz_ = 32;
  // lower bound for LR-consistency
  int tlo_ = 25;
  // upper bound for LR-consistency
  int thi_ = 100;
  // epipolar threshold (hamming distance)
  // for epipolar refinement matches
  int epiThres_ = 2;
  // do dense interpolation
  bool denseInterpolation_ = false;
  // debug information
  bool debug_ = false;
  // use final disparty filter
  bool useFinalFilter_ = true;
  
  FadeSettings(int niters, int szz, int tlo, int thi, int epiThres, 
      bool denseInterpolation, bool useFinalFilter, bool debug) :
    niters_(niters), szz_(szz), tlo_(tlo), thi_(thi), epiThres_(epiThres), 
    denseInterpolation_(denseInterpolation), useFinalFilter_(useFinalFilter), debug_(debug){}
  FadeSettings(){}
  // Builder methods
  FadeSettings& builder(void){
    return *this;
  }
  FadeSettings& niters(int niters){
    this->niters_ = niters;
    return *this;
  }
  FadeSettings& szz(int szz){
    this->szz_ = szz;
    return *this;
  }
  FadeSettings& tlo(int tlo){
    this->tlo_ = tlo;
    return *this;
  }
  FadeSettings& thi(int thi){
    this->thi_ = thi;
    return *this;
  }
  FadeSettings& epithres(int epithres){
    this->epiThres_ = epithres;
    return *this;
  }
  FadeSettings& denseInterpolation(bool denseInterpolation){
    this->denseInterpolation_ = denseInterpolation;
    return *this;
  }
  FadeSettings& debug(bool debug){
    this->debug_ = debug;
    return *this;
  }
  FadeSettings& useFinalFilter(bool useFinalFilter){
    this->useFinalFilter_ = useFinalFilter;
    return *this;
  }
};
struct DisparityResult {
  std::vector<Support> initialSupport;
  std::vector<Support> finalSupport;
  ndb::Buffer<uint8_t> cost;
  DisparityResult() {
  }
};
class Fade {
  public:
    Fade() {
    }
   /**
     * @brief      Draws a delaunay triangulation
     *
     * @param[in]  <unnamed>  { parameter_description }
     * @param[in]  filename   The filename
     */
    void drawDelaunay( std::vector<Support>& supp,
        std::vector<Triangle>& triangles,
        Buffer<uint8_t>& img,
        std::string filename) {
      //Copy the image
      Buffer<uint8_t> outImg = img;
      //Draw triangles
      for (int i = 0; i < triangles.size(); i++) {
        outImg.drawTriangle(supp[triangles[i].v1], supp[triangles[i].v2], supp[triangles[i].v3], 255);
      }
      outImg.writePNG(filename);
    }
   void drawDelaunay(std::vector<Support>& supp,  
        std::vector<Triangle>& triangles,
        Buffer<ndb::RGBColor>& img) {
      //Copy the image
      //Draw triangles
      for (int i = 0; i < triangles.size(); i++) {
        img.drawTriangle(supp[triangles[i].v1], supp[triangles[i].v2], 
            supp[triangles[i].v3], ndb::RGBColor(220,220,220));
      }
    }
    /**
     * @brief      Reads in a new pair of images to be processed.
     *
     * @param      simg  The source image
     * @param      timg  The target image
     * @param      filtermask a filtermask generated from the gpc forest used
     *
     * @return     a disparity estimate
     */
    DisparityResult disparityEstimate(
        gpc::inference::Forest::PreprocessedImage& simg,
        gpc::inference::Forest::PreprocessedImage& timg,
        gpc::inference::Forest::FilterMask& filtermask, 
        gpc::inference::InferenceSettings& gpcsettings,
        FadeSettings& fadesettings){

      //define forest
      typedef gpc::inference::Forest GPCForest_t;
      GPCForest_t forest;

      //We do not count reading the fastmask from forest as part of eval time.
      std::vector<int16_t> fastmask;
      std::vector<int8_t> taus;
      Dimension srcDim = simg.smooth.getDimension();


      DisparityResult result;

      result.initialSupport = forest.rectifiedMatch(simg, timg, filtermask, gpcsettings);
         
      std::vector<Support> supp = result.initialSupport;
      
      //Delauney triangulation
      std::vector<Triangle> triangles = delauneyMesh(supp);
      if (triangles.size() == 0) return result;
      //Save delauney triangulation image
      // drawDelaunay(supp, triangles, srcImg, "triangulation.png");
      
      //Grid size gets shrunk during interpoaltion iterations
      int szz = fadesettings.szz_; 

      int thres=255;
      if(fadesettings.useFinalFilter_ == true)
        thres=fadesettings.thi_;
      ndb::Buffer<float> Df(simg.smooth.rows(), simg.smooth.cols(),0);
      ndb::Buffer<uint8_t> Cf(simg.smooth.rows(), simg.smooth.cols(), thres); //init with some high cost

      for (int it = 0; it < fadesettings.niters_; it++) {
        //Assign each pixel within the triangulation the id of the triangle it belongs to.
        Buffer<uint16_t> pixelMap(Eigen::Vector2i(simg.smooth.cols(),simg.smooth.rows()), 65535);
        for (int i = 0; i < triangles.size(); i++) {
          pixelMap.fillTriangle(supp[triangles[i].v1], supp[triangles[i].v2], supp[triangles[i].v3], i);
        }

        //Plane parameter computation for each triangle
        std::vector<Eigen::Vector3f> planeParams = computePlaneParameters(triangles, supp);
        //Interpolate disparities (in high gradient regions that didn't receive a depth prior by gpc)

        ndb::Buffer<float> Ditt(simg.smooth.rows(),simg.smooth.cols());
        std::vector<int> validEstimateIndices;

        if (fadesettings.denseInterpolation_)
          Ditt = DenseDisparityInterpolation(pixelMap, planeParams, validEstimateIndices);
        else
          Ditt = SemiDenseDisparityInterpolation(pixelMap, simg.mask, planeParams, validEstimateIndices);

        //Fill interpolated disparities into ugly former GPC format
        std::vector<std::pair<ndb::Point, ndb::Point>> corr;
        //Cost evaluation
        Buffer<uint8_t> Citt;
        Citt = costEvaluationBoxDenseSSE(simg.smooth, timg.smooth, Ditt, validEstimateIndices);

        //visualize cost
        if (fadesettings.debug_) {
          Buffer<RGBColor> d = visualizeCost(simg.smooth, Citt);
          d.writePNGRGB("costVisualization_" + std::to_string(it) + ".png") ;
          drawDelaunay(supp, triangles, simg.smooth, "triangulation_" + std::to_string(it) + ".png");
        }

                //Disparity refinement
        Dimension dim = pixelMap.getDimension();
        Buffer<ConfidentSupport> Cg(Eigen::Vector2i(1 + dim.w / szz, 1 + dim.h / szz), 
            ConfidentSupport(0, 0, 0, fadesettings.tlo_));
        Buffer<InvalidMatch> Cb(Eigen::Vector2i(1 + dim.w / szz, 1 + dim.h / szz), 
            InvalidMatch(0, 0, fadesettings.thi_));

        disparityRefinement(dim, szz, it,  Ditt, Citt, Df, Cf, validEstimateIndices, Cg, Cb, fadesettings);
        if(fadesettings.debug_){
          Buffer<RGBColor> disp = ndb::getDisparityVisualization(simg.smooth, validEstimateIndices, Df);
          disp.writePNGRGB("disparity_" + std::to_string(it) + ".png");
        }
        
        //If not last iteration: Support resampling, retriangulate.
        if (it < fadesettings.niters_ - 1){ //resample supports if we're not done yet
          std::vector<ndb::Support> matches = supportResampling(simg.smooth, timg.smooth, simg.grad, timg.grad,
              pixelMap, triangles, fastmask, dim, Cg, Cb, supp,
              szz, fadesettings);


          //retriangulate
          triangles.clear();
          triangles = delauneyMesh(supp);

          szz = std::max(1, szz / 2);
        }
        if (it == fadesettings.niters_ - 1) {
          if (fadesettings.useFinalFilter_ == true) {
            result.cost = Cf;
            for (auto& idx : validEstimateIndices) {
              int x = idx % simg.smooth.cols();
              int y = idx / simg.smooth.cols();
              if (Cf.getPixel(x, y) < fadesettings.thi_) {
                result.finalSupport.push_back(Support(x, y, Df.getPixel(x, y)));
              }
            }
          } else {
            result.cost = Cf;
            for (auto& idx : validEstimateIndices) {
              int x = idx % simg.smooth.cols();
              int y = idx / simg.smooth.cols();
              //How the paper does it
              if (Cf.getPixel(x, y) < 255) {
                result.finalSupport.push_back(Support(x, y, Df.getPixel(x, y)));
              }
            }
          }
 
          return result;
        }
      }
      return result;
    }
private:
    /**
     * @brief      Delauney Mesh using the triangle library This method has been
     *             adapted from elas.cpp
     *
     * @return     { description_of_the_return_value }
     */
    std::vector<Triangle> delauneyMesh(std::vector<Support>& supp) {
      struct triangulateio in, out;
      //sanity check: abort if we have less than 3 points
      if (supp.size() < 3) {
        std::vector<Triangle> tri;
        return tri;
      }
      int32_t k;
      // inputs
      in.numberofpoints = supp.size();
      in.pointlist = (float*)malloc(in.numberofpoints * 2 * sizeof(float));
      k = 0;

      for (int32_t i = 0; i < supp.size(); i++) {
        in.pointlist[k++] = supp[i].x;
        in.pointlist[k++] = supp[i].y;
      }


      in.numberofpointattributes = 0;
      in.pointattributelist      = NULL;
      in.pointmarkerlist         = NULL;
      in.numberofsegments        = 0;
      in.numberofholes           = 0;
      in.numberofregions         = 0;
      in.regionlist              = NULL;

      // outputs
      out.pointlist              = NULL;
      out.pointattributelist     = NULL;
      out.pointmarkerlist        = NULL;
      out.trianglelist           = NULL;
      out.triangleattributelist  = NULL;
      out.neighborlist           = NULL;
      out.segmentlist            = NULL;
      out.segmentmarkerlist      = NULL;
      out.edgelist               = NULL;
      out.edgemarkerlist         = NULL;

      // do triangulation (z=zero-based, n=neighbors, Q=quiet, B=no boundary markers)
      char parameters[] = "zQB";
      triangulate(parameters, &in, &out, NULL);
      // put resulting triangles into vector tri

      std::vector<Triangle> tri;
      k = 0;
      for (int32_t i = 0; i < out.numberoftriangles; i++) {
        tri.push_back(Triangle(out.trianglelist[k], out.trianglelist[k + 1], out.trianglelist[k + 2]));
        k += 3;
      }

      // free memory used for triangulation
      free(in.pointlist);
      free(out.pointlist);
      free(out.trianglelist);

      // return triangles
      return tri;
    }

  /**
   * @brief      Calculates the plane parameters for each triangle.
   *
   * @param      tria  The triangles as a set  triplets indexing (x,y,d) 
   *                   triplets in the suppor
   * @param      supp  The support
   *
   * @return     The plane parameters.
   */
  std::vector<Eigen::Vector3f> computePlaneParameters(
      std::vector<Triangle>& tria,
      std::vector<Support>& supp) {
    std::vector<Eigen::Vector3f> planeParams(tria.size());
    for (int i = 0; i < tria.size(); i++ ) {

      Eigen::Matrix3f A;
      A <<  supp[tria[i].v1].x, supp[tria[i].v1].y, 1,
        supp[tria[i].v2].x, supp[tria[i].v2].y, 1,
        supp[tria[i].v3].x, supp[tria[i].v3].y, 1;

      Eigen::Vector3f b;

      b << supp[tria[i].v1].d, supp[tria[i].v2].d, supp[tria[i].v3].d;
      planeParams[i] = (A.partialPivLu().solve(b));
    }
    return planeParams;
  }

  /**
   * @brief      Interpolates disparity for all high gradient pixels based on
   *             graph given by delauney triangulation and sparse supports
   *             (disparity values by GPC)
   *
   * @param      pixelMap     The pixel map
   * @param      srcGrad      The source gradient
   * @param      planeParams  The plane parameters
   *
   * @return     Supports
   */
 ndb::Buffer<float> SemiDenseDisparityInterpolation(
      ndb::Buffer<uint16_t>& pixelMap,
      std::vector<int>& srcGrad1D,
      std::vector<Eigen::Vector3f>& planeParams,
      std::vector<int>& validEstimateIndices) {

    Dimension dim = pixelMap.getDimension();
    ndb::Buffer<float> candidateDisparities(dim.h, dim.w);
    uint16_t* ptr = pixelMap.data();
    int dMin = 0, dMax = 0;
    float dAvg = 0.f;
    //high gradient pixels
    for (int i = 0; i < srcGrad1D.size(); i++) {
      //Corresponding triangle ID
      uint16_t triaId = ptr[srcGrad1D[i]];

      int x = srcGrad1D[i] % pixelMap.cols();
      int y = srcGrad1D[i] / pixelMap.cols();
      if (triaId != 65535) { //only allow gradient pixels that have a triangle assignment
        float d = planeParams[triaId](0) * x + planeParams[triaId](1) * y + planeParams[triaId](2);
        candidateDisparities.setPixel(x, y, d);
        validEstimateIndices.push_back(srcGrad1D[i]);
      }
    }
    return candidateDisparities;
  }

  /**
   * @brief      Dense disparity interpolation. Uses all pixels within the
   *             initially established convex polytope outlined by the Delauney
   *             triangulation
   *
   * @param[in]  pixelMap              The pixel map
   * @param[in]  planeParams           The plane parameters for each triangle
   *                                   within the map
   * @param[out] validEstimateIndices  The indices within the returned disparity
   *                                   buffer that have received a disparity
   *                                   estimate.
   *
   * @return     disparity buffer the size of the image
   */
  ndb::Buffer<float> DenseDisparityInterpolation(
      ndb::Buffer<uint16_t>& pixelMap,
      std::vector<Eigen::Vector3f>& planeParams,
      std::vector<int>& validEstimateIndices) {
    Dimension dim = pixelMap.getDimension();
    ndb::Buffer<float> candidateDisparities(dim.h, dim.w);
    int dMin = 0, dMax = 0;
    float dAvg = 0.f;
    //All pixels
    for (int x = 0; x < dim.w; x++)
      for (int y = 0; y < dim.h; y++) {
        //Corresponding triangle ID
        uint16_t triaId = pixelMap.getPixel(x, y);

        if (triaId != 65535) { //exclude pixels outside of the triangulation
          float d = planeParams[triaId](0) * x + planeParams[triaId](1) * y + planeParams[triaId](2);
          candidateDisparities.setPixel(x, y, d);
          validEstimateIndices.push_back(y * dim.w + x);
        }
      }
    return candidateDisparities;
  }
  Buffer<uint8_t> costEvaluationBoxDenseSSE(
      Buffer<uint8_t>& srcImg,
      Buffer<uint8_t>& tarImg,
      Buffer<float>& candidateDisparities,
      std::vector<int>& validEstimateIndices){
    //Census images of source and target
    ndb::Buffer<uint32_t> censusSrc(srcImg.rows(),srcImg.cols());
    ndb::Buffer<uint32_t> censusTar(srcImg.rows(),srcImg.cols());
    ndb::census5x5(srcImg.data(), censusSrc.data(), srcImg.cols(), srcImg.rows());
    ndb::census5x5(tarImg.data(), censusTar.data(), tarImg.cols(), tarImg.rows());
    //Match cost that we are returning
    ndb::Buffer<uint8_t> matchCost(srcImg.rows(), srcImg.cols(), 255);
    int minCost = 0., maxCost = 0., totCost = 0.;
    //Copy pixels pertinent to calculation from buffer
    // This could probably also be done with SSE
    int cnt=0;
    for (auto idx : validEstimateIndices) {
      int xSrc = idx % srcImg.cols();
      int ySrc = idx / srcImg.cols();
      float d = candidateDisparities.getPixel(xSrc, ySrc);
      int xTar = xSrc - d;
      //get census transforms
      uint32_t cSrc = censusSrc.getPixel(xSrc, ySrc);
      uint32_t cTar = censusTar.getPixel(xTar, ySrc);

      uint32_t diff = cSrc ^ cTar;
      uint8_t cost = uint8_t(_mm_popcnt_u32(diff)) ;
      matchCost.setPixel(xSrc, ySrc, cost);
      totCost += cost;
    }
    return matchCost;

  }
  /**
   * @brief      Visualizes the cost in red
   *
   * @param      srcImg  The source image, where the cost will be drawn onto
   * @param      cand    The cand
   * @param[in]  cost    The cost for each candidate disparity
   *
   * @return     { description_of_the_return_value }
   */
  Buffer<RGBColor> visualizeCost(
      Buffer<uint8_t>& srcImg,
      Buffer<uint8_t>& cost) {
      Buffer<RGBColor> errors(Eigen::Vector2i(cost.width, cost.rows()));

    //Copy image into a three channel color image first:
    for (int x = 0; x < srcImg.cols(); x++) {
      for (int y = 0; y < srcImg.rows(); y++) {
        uint8_t p = srcImg.getPixel(x, y);
        errors.setPixel(x, y, RGBColor(p, p, p));
        //Overwrite pixel in red if we have significant error.
        uint8_t err = cost.getPixel(x, y);
        if(err != 255)
          errors.setPixel(x, y, RGBColor(min(255,err*16), 0, 0));
      }

    }
    return errors;
  }
  /**
   * @brief      Refine disparity
   *
   * @param      dim    The dimension of the image
   * @param[in]  szocc  The occupancy grid.
   * @param      Dit    The disparacy estimate
   * @param      Cit    The cost of this estimate
   * @param      Df     The current final (filtered) disparity
   * @param      Cf     The cost of the filtered disparity
   * @param      Cg     { parameter_description }
   * @param      Cb     { parameter_description }
  */
  void disparityRefinement(
        Dimension & dim,
        int szocc,
        int numIt,
        ndb::Buffer<float>& Dit,
        ndb::Buffer<uint8_t>& Cit, 
        ndb::Buffer<float>& Df,
        ndb::Buffer<uint8_t>& Cf,
        std::vector<int>& validEstimateIndices,
        Buffer<ConfidentSupport>& Cg,
        Buffer<InvalidMatch>& Cb,
        FadeSettings& fadesettings) {
    //Occupancy grid
    int H = dim.h / szocc;
    int W = dim.w / szocc;

    for (auto& idx : validEstimateIndices) {
      //Pixel coords of disparity value
      int x = idx % dim.w;
      int y = idx / dim.w;

      //Refinement grid cells
      int xdash = x / szocc;
      int ydash = y / szocc;

      int cit = Cit.getPixel(x, y);
      float dit = Dit.getPixel(x, y);

      //paper does the following:
      if ( cit < Cf.getPixel(x, y)) {
        Df.setPixel(x, y, dit );
        Cf.setPixel(x, y, cit);
      }
      //update pixel with lowest cost in this grid cell
      if (cit < fadesettings.tlo_ && cit < Cg.getPixel(xdash, ydash).cost) {
        Cg.setPixel(xdash, ydash, ConfidentSupport(x, y, dit, cit));
      }
      //update pixel with highest cost in this grid cell
      if (cit > fadesettings.thi_ && cit > Cb.getPixel(xdash, ydash).cost) {
        Cb.setPixel(xdash, ydash, InvalidMatch(x, y, cit));
      }
    }
  }

  /**
   * @brief      { function_description }
   *
   * @param      Cg    { parameter_description }
   * @param      Cb    { parameter_description }
   * @param      supp  The supp
   */
  std::vector<ndb::Support> supportResampling(
      Buffer<uint8_t>& srcImg,
      Buffer<uint8_t>& tarImg,
      Buffer<uint8_t>& srcGrad,
      Buffer<uint8_t>& tarGrad,
      Buffer<uint16_t>& pixelMap,
      std::vector<Triangle>& triangles,
      std::vector<int16_t>& fastmask,
      Dimension & dim,
      Buffer<ConfidentSupport>& Cg,
      Buffer<InvalidMatch>& Cb,
      std::vector<Support>& supp,
      int szocc,
      FadeSettings& fadesettings){
    std::vector<ndb::Point> X;
    int H = dim.h / szocc;
    int W = dim.w / szocc;
    //Iterate over grid cells.
    //Lowest cost point per grid cell -> support
    //Highest cost point per grid cell -> epipolar matching -> support
    for (int u = 0; u < W; u++) {
      for (int v = 0; v < H ; v++) {
        //Collect points with high cost for epipolar matching
        if (Cb.getPixel(u, v).cost > fadesettings.thi_) {
          X.push_back(ndb::Point(Cb.getPixel(u, v).x, Cb.getPixel(u, v).y)); //add supports with disparity 0
        }
        //Collect points with lowest cost and add to support if within triangle bounds
        if (Cg.getPixel(u, v).cost < fadesettings.tlo_) {
          int x = Cg.getPixel(u, v).x;
          int y = Cg.getPixel(u, v).y;
          float d = Cg.getPixel(u, v).d;
          supp.push_back(ndb::Support(x, y, d));
        }
      }
    }

    //Epipolar census match
    std::vector<ndb::Support> matches =  EpipolarBlockMatch(srcImg, tarImg, tarGrad, X, pixelMap, 
        triangles, supp, fadesettings);

    //Added for interpolation / cost evaluation on next iteration
    for (auto m : matches) 
      supp.push_back(Support(m.x, m.y, m.d));
      
    return matches;
  }

  /**
   * @brief      Epipolar matching based on block matching with 5x5 neighborhoods.
   *             Assumption: Input images are rectified, i.e. epipolar lines are
   *             horizontal and features are to be found in target image at same
   *             vertical position as in source.
   *
   * @param      src        The source image
   * @param      tar        The target image
   * @param      tarGrad    The gradient of the target image.
   * @param      srcPoints  A set of points in the source image
   * @param[in]  dispL      lower bound for disparity
   * @param[in]  dispH      upper bound for disparity
   * @param[in]  threshold  The error threshold to classify a candidate as a
   *                        match.
   *
   * @return     A set of supports (x,y,d) based on correspondences that were
   *             found
  */
  std::vector<ndb::Support>
    EpipolarBlockMatch(
    Buffer<uint8_t>& src,
    Buffer<uint8_t>& tar,
    Buffer<uint8_t>&  tarGrad,
    std::vector<ndb::Point>& srcPoints,
    Buffer<uint16_t>& pixelMap,
    std::vector<Triangle>& triangles,
    std::vector<Support>& supp,
    FadeSettings& fadesettings) {
          
    std::vector<ndb::Support> out;
    //For all candidate points in the source image
    for (auto& sp : srcPoints) {
      int minCost = 10000;
      int dispOfMinimum;
      int numMatches = 0;
      int triaID = pixelMap.getPixel(sp.x, sp.y);
      if (triaID != 0xFFFF) { //valid pixel (i.e. within triangulation)
        int v1 = triangles[triaID].v1;
        int v2 = triangles[triaID].v2;
        int v3 = triangles[triaID].v3;
        Eigen::MatrixXf A(3, 3);
        A << 1, 1, 1, supp[v1].x , supp[v2].x , supp[v3].x , supp[v1].y, supp[v2].y , supp[v3].y;
        float area = A.determinant();

        //minimum and maximum disparity within these triangles
        float minTriaDisp = min(supp[v1].d, min(supp[v2].d, supp[v3].d));
        float maxTriaDisp = max(supp[v1].d, max(supp[v2].d, supp[v3].d));

        //Scan given disparity range given by triangle
        for (int ix = sp.x - maxTriaDisp; ix < sp.x - minTriaDisp; ix++) {
          //Evaluate only high gradient target pixels if semidense, skip rest.
          if (fadesettings.denseInterpolation_ == true
            || (fadesettings.denseInterpolation_ == false && tarGrad.getPixel(ix, sp.y) == 255)) {
            int hammingDist = 0;
            //Values of src and target patches
            uint8_t srcCenter = src.getPixel(sp.x, sp.y);
            uint8_t tarCenter = tar.getPixel(ix, sp.y);
            for (int x =  - 4; x <= 4; x += 2) {
              for (int y = - 4; y <= 4; y += 2) {
                if ((src.getPixel(sp.x + x, sp.y + y) > srcCenter)
                  != (tar.getPixel(ix + x, sp.y + y) > tarCenter))
                    hammingDist++;
               }
            }
            //target neighborhood
            int d = sp.x - ix;
            if (hammingDist < fadesettings.epiThres_//matching threshold
              && hammingDist <= minCost) { //better or equal cost to previous
              minCost = hammingDist;
              dispOfMinimum = d;
              numMatches++;
             }
          }//if high gradeint
        }//ix for loop
        //If we found a single(!) match
        if (numMatches == 1) {
          out.push_back(ndb::Support(sp.x, sp.y,  dispOfMinimum));
        }
      }//if valid pixel
    }//for all source points
    return out;
  }//EpipolarBlockMatch
};//fade class 
}//ndb namespace
#endif
