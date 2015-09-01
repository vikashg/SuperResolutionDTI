/*
 * STKTensorEstimation_v2.cpp
 *
 *  Created on: Aug 17, 2015
 *      Author: vgupta
 */


#include "itkImageSeriesReader.h"
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
#include "itkExtractImageFilter.h"

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



    const string  file_n = cl.follow("NoFile",1, "-i");
    const string  file_g = cl.follow("NoFile",1, "-g");
    const string  mask_n = cl.follow("NoFile",1, "-m");

    const int  numOfB0s = cl.follow(1, 1, "-n");

    typedef float RealType;
    typedef itk::Image<RealType, 4> Scalar4DImageType;
    typedef itk::ImageFileReader<Scalar4DImageType> Reader4DType;

    Reader4DType::Pointer reader4DScalar = Reader4DType::New();
    reader4DScalar->SetFileName(file_n.c_str());
    reader4DScalar->Update();
    Scalar4DImageType::Pointer dwiImageList = reader4DScalar->GetOutput();

    Scalar4DImageType::SizeType size4d;
    size4d = dwiImageList->GetLargestPossibleRegion().GetSize();



    //Convert the 4D image to a vector of images
    int ImageDim = 3;
    typedef itk::Image<RealType, 3> ScalarImageType;
    typedef std::vector<ScalarImageType::Pointer> ImageListType;
    ImageListType DWIList;
    int numOfImages = size4d[3];

    typedef itk::ExtractImageFilter<Scalar4DImageType, ScalarImageType> ExtractImageFilterType;
    ExtractImageFilterType::Pointer extractImageFilter = ExtractImageFilterType::New();

    typedef itk::ImageFileWriter<ScalarImageType> WriterType;

    for (int i=0; i < numOfImages; i++)
    {
        ExtractImageFilterType::Pointer extractImageFilter = ExtractImageFilterType::New();
        Scalar4DImageType::SizeType size;
        Scalar4DImageType::IndexType index;
        index[0]= 0; index[1]=0; index[2]=0; index[3]=i;

        size[0]=size4d[0]; size[1]=size4d[1]; size[2]= size4d[2]; size[3]=0;
        Scalar4DImageType::RegionType region(index, size);

        extractImageFilter->SetDirectionCollapseToIdentity();

        extractImageFilter->SetInput(dwiImageList);
        extractImageFilter->SetExtractionRegion(region);
        extractImageFilter->Update();

        WriterType::Pointer writer = WriterType::New();

        		int num =i;
        		std::ostringstream num_con;
        		num_con << num;
        		std::string result  = num_con.str() + ".nii.gz";

        		writer->SetFileName(result); writer->SetInput(extractImageFilter->GetOutput());
        		writer->Update();

    }






    return 0;
}
