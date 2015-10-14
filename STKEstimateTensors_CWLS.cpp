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
#include "WeightedLeastSquares_CWLS.h"
#include "TensorUtilites.h"

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

 // Usual Typedefs
    typedef float RealType;
    const int ImageDim  =3;

    typedef itk::Image<RealType, ImageDim> ScalarImageType;
    typedef itk::Vector<double, ImageDim> VectorType;
    typedef itk::Image<VectorType, ImageDim> VectorImageType;
    typedef itk::ImageFileReader<ScalarImageType> ScalarFileReaderType;


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


 //Read Mask
    ScalarFileReaderType::Pointer maskReader = ScalarFileReaderType::New();

    maskReader->SetFileName(mask_n.c_str());
    maskReader->Update();
    ScalarImageType::Pointer maskImage_HR = maskReader->GetOutput();


//Read B0 Image
    ScalarFileReaderType::Pointer B0Reader = ScalarFileReaderType::New();
    B0Reader->SetFileName(B0_n.c_str());
    B0Reader->Update();

    ScalarImageType::Pointer B0Image_HR = B0Reader->GetOutput();

	
    //Now do a retarded tensor estimation

   UnweightedLeastSquaresTensorEstimation TensorEstimation;
   TensorEstimation.ReadDWIList(DWIList);
   TensorEstimation.ReadGradientList(GradientList);
   TensorEstimation.ReadMask(maskImage_HR);
   TensorEstimation.ReadBVal(1.0);
   TensorEstimation.ReadB0Image(B0Image_HR);

   typedef itk::DiffusionTensor3D<RealType> DiffusionTensorType;
   typedef itk::Image<DiffusionTensorType, 3> TensorImageType;

 /*  TensorImageType::Pointer tensorImage = TensorEstimation.Compute();

	std::cout << "TensorEstimation Done " << std::endl;

   	typedef itk::ImageFileWriter<TensorImageType> TensorImageWriterType;
   	TensorImageWriterType::Pointer tensorImageWriter = TensorImageWriterType::New();
	

	TensorUtilities utilsTensors;

	TensorImageType::Pointer tensorImage_removeNans = utilsTensors.ReplaceNaNsReverseEigenValue(tensorImage, maskImage_HR);
	
	  TensorImageType::Pointer log_tensorImage = utilsTensors.LogTensorImageFilter(tensorImage_removeNans, maskImage_HR);

	                 TensorImageType::Pointer removed_nans_logTensorImage = utilsTensors.ReplaceNaNsInfs(log_tensorImage, maskImage_HR);

	TensorImageType::Pointer padded_tensorImage_init = utilsTensors.ExpTensorImageFilter(removed_nans_logTensorImage, maskImage_HR);
	
	tensorImageWriter->SetFileName("tensorImage.nii.gz");
	tensorImageWriter->SetInput(padded_tensorImage_init);
	tensorImageWriter->Update();
*/

	typedef itk::ImageFileReader<TensorImageType> TensorReaderType;
	TensorReaderType::Pointer tensorReader = TensorReaderType::New();

	tensorReader->SetFileName("tensorImage.nii.gz");
	tensorReader->Update();
	TensorImageType::Pointer padded_tensorImage_init = tensorReader->GetOutput();


	WeightedLeastSquares wls;

	wls.ReadDWIListHR(DWIList);
	wls.ReadGradientList(GradientList);
	wls.ReadB0ImageHR(B0Image_HR);
	wls.ReadHRMask(maskImage_HR);
	wls.ReadBVal(1);	
//	wls.ComputeDelSim(padded_tensorImage_init);
	wls.ReadTensorImage(padded_tensorImage_init);
	wls.UpdateTerms();

/*	typedef itk::Image<TensorImageType> TensorImageWriterType;	
	TensorImageWriterType::Pointer tensorWriter = TensorImageWriterType::New();
	tensorWriter->SetFileName("estimatedTensors.nii.gz");
	tensorWriter->SetInput(estimatedTensor);
	tensorWriter->Update();
*/
	return 0;
}
