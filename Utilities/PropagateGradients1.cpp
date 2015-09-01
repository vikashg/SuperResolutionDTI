/*
 * PropagateGradients.cxx
 *
 *  Created on: Jul 26, 2015
 *      Author: vgupta
 */


#include "iostream"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "../GetPot/GetPot"
#include "itkDiffusionTensor3D.h"

#include "itkImageRegionIterator.h"
#include "itkVector.h"
#include "itkMatrix.h"
#include "vnl/vnl_matrix.h"
#include "vnl/vnl_inverse.h"

#include "../inc/UnweightedLeastSquaresTensorFit.h"

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

    const string grad_file_N = cl.follow("NoFile", 1, "-g");
    const string ref_file = cl.follow("NoFile",1,"-r");

    typedef float RealType;
    typedef itk::Vector<RealType, 3> VectorType;
    typedef itk::Image<VectorType, 3> VectorImageType;
    typedef std::vector<VectorImageType::Pointer> VectorImageListType;

    typedef itk::ImageFileWriter<VectorImageType> WriterType;

    typedef itk::Image<RealType, 3> ImageType;
    typedef itk::ImageFileReader<ImageType> ReaderType;
    ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName(ref_file);
    reader->Update();

    ImageType::Pointer refImage = reader ->GetOutput();

    typedef std::vector<VectorType> GradientListType;
    GradientListType GradientList;
     std::ifstream fileg(grad_file_N);
     int numOfGrads = 0;
     fileg >> numOfGrads;

     for (int i=0; i < numOfGrads; i++)
      {
          VectorType g;
          fileg >> g[0]; fileg >>g[1]; fileg >> g[2];
          if (g!=0.0)
          {
              g.Normalize();
          }
          GradientList.push_back(g);
//            std::cout << g << std::endl;
      }


//     VectorImageType::IndexType testIndex;
//     testIndex[0]=128; testIndex[1]=128; testIndex[2]=15;

     for (int i=0; i < numOfGrads; i++)
     {
    	 VectorImageType::Pointer image = VectorImageType::New();
    	 image->SetDirection(refImage->GetDirection());
    	 image->SetOrigin(refImage->GetOrigin());
    	 image->SetSpacing(refImage->GetSpacing());
    	 image->SetRegions(refImage->GetLargestPossibleRegion());
    	 image->Allocate();
    	 image->FillBuffer(GradientList[i]);

    	    WriterType::Pointer writer = WriterType::New();

    	    std::ostringstream c;
    	    c<< i;

    	   std::string _C_str;
    	   _C_str=c.str() ;
    	   std::string tempName;
    	   tempName = "Gradient_" + _C_str + ".nii.gz";

//    	   std::cout << image->GetPixel(testIndex) << std::endl;
    	   std::cout << "Grad " << GradientList[i] << std::endl;


    	    writer->SetFileName(tempName);
    	    writer->SetInput(image);
    	    writer->Update();

     }


     return 0;
}


