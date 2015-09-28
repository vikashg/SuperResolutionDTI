/*
 * STKEstimateTensors.cxx
 *
 *  Created on: Jul 24, 2015
 *      Author: vgupta
 */


#include "iostream"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "GetPot/GetPot"
#include "inc/MapFilterLR2HR.h"
#include "itkDiffusionTensor3D.h"
#include "JointTensorEstimation.h"
#include "CopyImage.h"
#include "../inc/UnweightedLeastSquaresTensorFit.h"
#include "ComputeSigma.h"
#include "itkImageRegionIterator.h"
#include "TotalEnergy.h"
#include "vnl/vnl_matrix_exp.h"


using namespace std;

int main (int argc, char *argv[])
{

    GetPot cl (argc, const_cast<char**>(argv));
    if( cl.size() == 1 || cl.search (2,"--help","-h") )
    {
        std::cout << "Not Enough Arguments" << std::endl;
        std::cout << "Scales the tensors with a scalar factor" << std::endl;
        std::cout << "Usage: -trueB0 <true B0> -m <MaskImage> -true <True Tensors> -f <flag for extended gradient> -t <initial tensor estimate> -g <gradient> -o <Output File> -s <Sigma> -nm <Noise Model> -Sim <intelligent COnvergence>" << std::endl;
      return -1;
    }




     const string file_g_n = cl.follow("NoFile",1, "-g");
     const string file_in  = cl.follow("NoFile",1,"-i");
     const string B0_n     =  cl.follow("NoFile", 1, "-B0");
     const string mask_n   = cl.follow("NoFile",1, "-m");
     const int numOfIter   = cl.follow(1,1, "-n");
     const float kappa_L   = cl.follow(0.05, 1, "-k");
     const float lambda_L  = cl.follow(0.25, 1, "-lamb_L");


     std::cout << file_g_n << std::endl;
    
    // Usual Typedefs
    typedef float RealType;
    const int ImageDim  =3;

    typedef itk::Image<RealType, ImageDim> ScalarImageType;
    typedef itk::Vector<double, ImageDim> VectorType;
    typedef itk::Image<VectorType, ImageDim> VectorImageType;


    //Read Mask
    typedef itk::ImageFileReader<ScalarImageType> ScalarFileReaderType;
    ScalarFileReaderType::Pointer maskReader = ScalarFileReaderType::New();

    maskReader->SetFileName(mask_n.c_str());
    maskReader->Update();
    ScalarImageType::Pointer maskImage = maskReader->GetOutput();


    ScalarFileReaderType::Pointer B0Reader = ScalarFileReaderType::New();
    B0Reader->SetFileName(B0_n.c_str());
    B0Reader->Update();

    ScalarImageType::Pointer B0Image = B0Reader->GetOutput();

    std::cout << "Read mask " << std::endl;

    // Read GradientFiles
    typedef itk::ImageFileReader<VectorImageType> VectorFileReaderType;
    typedef std::vector<VectorImageType::Pointer> VectorImageListType;
    VectorImageListType GradientList;

    std::ifstream file_g(file_g_n.c_str());
    int numOfImages = 0;
    file_g >> numOfImages;

    for (int i=0; i < numOfImages  ; i++) // change of numOfImages
      {
          char filename[256];
          file_g >> filename;
          VectorFileReaderType::Pointer myReader=VectorFileReaderType::New();
          myReader->SetFileName(filename);
          std::cout << "Reading.." << filename << std::endl; // add a try catch block
          myReader->Update();
          GradientList.push_back( myReader->GetOutput() ); //using push back to create a stack of diffusion images
      }

    // Finished Reading Gradient Files

    //Read DiffusionImages

    typedef std::vector<ScalarImageType::Pointer> ImageListType;
    ImageListType DWIList;

    std::ifstream file(file_in.c_str());
    int numOfImages_1 = 0;
    file >> numOfImages_1;

    for (int i=0; i < numOfImages_1  ; i++) // change of numOfImages
      {
          char filename[256];
          file >> filename;
          ScalarFileReaderType::Pointer myReader=ScalarFileReaderType::New();
          myReader->SetFileName(filename);
          std::cout << "Reading.." << filename << std::endl; // add a try catch block
          myReader->Update();
          DWIList.push_back( myReader->GetOutput() ); //using push back to create a stack of diffusion images
      }


    //Now do a retarded tensor estimation

   UnweightedLeastSquaresTensorEstimation TensorEstimation;
   TensorEstimation.ReadDWIList(DWIList);
   TensorEstimation.ReadGradientList(GradientList);
   TensorEstimation.ReadMask(maskImage);
   TensorEstimation.ReadBVal(1.0);
   TensorEstimation.ReadB0Image(B0Image);

   typedef itk::DiffusionTensor3D<RealType> DiffusionTensorType;
   typedef itk::Image<DiffusionTensorType, 3> TensorImageType;

   TensorImageType::Pointer tensorImage = TensorEstimation.Compute();
   typedef itk::ImageFileWriter<TensorImageType> TensorWriter;
   TensorWriter::Pointer writer = TensorWriter::New();

   TensorUtilities utils;
//   TensorImageType::Pointer dt_npt = utils.ReplaceNaNsReverseEigenValue(tensorImage, maskImage);
   writer->SetFileName("tensorImage_stupid.nii.gz");
   writer->SetInput(tensorImage);
   writer->Update();

   // Here compute the map





/*
   TensorImageType::Pointer logTensorImage = utils.LogTensorImageFilter(dt_npt, maskImage);

//
   vnl_vector<RealType> Sigma; Sigma.set_size(DWIList.size());
   Sigma.fill(1.0);
   ComputeSigma computeSigma;
   computeSigma.ReadDWIList(DWIList);
   computeSigma.ReadGradientList(GradientList);
   computeSigma.ReadMaskImage(maskImage);
   computeSigma.ReadB0Image(B0Image);
   computeSigma.ReadBVal(1.0);
   computeSigma.ReadTensorImage(tensorImage);

//   Sigma = computeSigma.ComputeAttenuation();
   Sigma = computeSigma.ComputeAttenuation_Frac();


   TensorImageType::IndexType testIndex;
   testIndex[0]=54; testIndex[1]=21; testIndex[2]=1;

//   std::cout << "D _index " << logTensorImage->GetPixel(testIndex) << std::endl;



   std::cout << Sigma << std::endl;

//
   JointTensorEstimation jTestimation;
   jTestimation.ReadBVal(1.0);
   jTestimation.ReadDWIList(DWIList);
   jTestimation.ReadB0Image(B0Image);
   jTestimation.ReadGradientList(GradientList);
   jTestimation.ReadInitialTensorImage(dt_npt);
   jTestimation.ReadKappa(kappa_L);
   jTestimation.ReadMaskImage(maskImage);
   jTestimation.ReadSigma(Sigma);
   jTestimation.ReadStepSize(0.0001);
   jTestimation.ReadLamba(lambda_L);
   jTestimation.ReadNumOfIterations(numOfIter);

   std::cout << "Update Terms" << std::endl;


   TensorImageType::Pointer estimatedTensorsLog = jTestimation.UpdateTerms();
   TensorImageType::Pointer estimatedTensors = utils.ExpTensorImageFilter(estimatedTensorsLog, maskImage);
   writer->SetFileName("estimatedtensorImage.nii.gz");
   writer->SetInput(estimatedTensors);
   writer->Update();

*/
   return 0;

}


