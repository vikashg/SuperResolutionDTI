/*
 * ReplaceB0CutOff.cxx
 *
 *  Created on: Aug 3, 2015
 *      Author: vgupta
 */

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIterator.h"
#include "../GetPot/GetPot"
#include "itkNeighborhoodIterator.h"
#include "CopyImage.h"

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

    const string B0_n = cl.follow("NoFile",1, "-i");
    const string mask_n  = cl.follow ("NoFile",  1 , "-m");
    const string out_n = cl.follow("NoFile", 1, "-o");
    const float cut_Off = cl.follow(100, 1 , "-c");

    typedef float RealType;
    typedef itk::Image<RealType, 3> ImageType;
    typedef itk::ImageFileReader<ImageType> ReaderType;

    ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName(B0_n);
    reader->Update();

    ImageType::Pointer B0_image = reader->GetOutput();

    ReaderType::Pointer maskReader = ReaderType::New();
    maskReader->SetFileName(mask_n);
    maskReader->Update();

    ImageType::Pointer maskImage = maskReader->GetOutput();

    typedef itk::ImageRegionIterator<ImageType> ScalarIterator;
    ScalarIterator itMask(maskImage, maskImage->GetLargestPossibleRegion());
    ScalarIterator itB0(B0_image, B0_image->GetLargestPossibleRegion());

    ImageType::SizeType radius; radius.Fill(1);
    typedef itk::NeighborhoodIterator<ImageType> NeighImageIterator;

    NeighImageIterator itB0Neigh(radius, B0_image, B0_image->GetLargestPossibleRegion());

    ImageType::Pointer outImage = ImageType::New();
    CopyImage cpImage;
    cpImage.CopyScalarImage(B0_image, outImage);
    ScalarIterator itOut(outImage, outImage->GetLargestPossibleRegion());


    for (itMask.GoToBegin(), itB0Neigh.GoToBegin(), itB0.GoToBegin(), itOut.GoToBegin();
    		!itMask.IsAtEnd(), !itB0.IsAtEnd(), !itB0Neigh.IsAtEnd(), !itOut.IsAtEnd();
    		++itMask, ++itB0, ++itB0Neigh, ++itOut)
    {
    	if (itMask.Get() != 0)
    	{
    		if (itB0.Get() < cut_Off)
    		{

    		RealType temp;
    		temp = (itB0Neigh.GetNext(0) + itB0Neigh.GetPrevious(0) + itB0Neigh.GetNext(1) + itB0Neigh.GetPrevious(1) +
    				itB0Neigh.GetNext(2) + itB0Neigh.GetPrevious(2))/6;

    		itOut.Set(temp);
    		}
    		else
    		{
    			itOut.Set(itB0Neigh.GetCenterPixel());
    		}
    	}
    }

    typedef itk::ImageFileWriter<ImageType> WriterType;
    WriterType::Pointer writer = WriterType::New();
    writer->SetFileName(out_n);
    writer->SetInput(outImage);
    writer->Update();

    return 0;
}



