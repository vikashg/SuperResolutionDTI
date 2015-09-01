/*
 * ConvertTTK2FSL.cpp
 *
 *  Created on: Aug 15, 2015
 *      Author: vgupta
 */

#include "itkDiffusionTensor3D.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "../GetPot/GetPot"
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

    const string dt_file1 = cl.follow("NoFIle", 1,  "-dt1");
    const string mask_file = cl.follow("NoFIle", 1,  "-m");

    typedef float RealType;
    typedef itk::DiffusionTensor3D<RealType> DiffusionType;
    typedef itk::Image<DiffusionType, 3> TensorImageType;

    typedef itk::Image<RealType, 3> ScalarImageType;

    typedef itk::ImageFileReader<TensorImageType> TensorReaderType;
    typedef itk::ImageFileReader<ScalarImageType> ScalarReaderType;

    TensorReaderType::Pointer tensorReader1 = TensorReaderType::New();

    tensorReader1->SetFileName(dt_file1.c_str());
    tensorReader1->Update();

    TensorImageType::Pointer tensorImage1 = tensorReader1->GetOutput();

    ScalarReaderType::Pointer scalarReader = ScalarReaderType::New();
    scalarReader->SetFileName(mask_file.c_str());
    scalarReader->Update();
    ScalarImageType::Pointer maskImage = scalarReader->GetOutput();



    typedef itk::ImageRegionIterator<TensorImageType> TensorIterator;
    typedef itk::ImageRegionIterator<ScalarImageType> ScalarIterator;
    TensorIterator itTens(tensorImage1, tensorImage1->GetLargestPossibleRegion());
    ScalarIterator itMask(maskImage, maskImage->GetLargestPossibleRegion());


    TensorImageType::IndexType testIndex;
    testIndex[0]=134; testIndex[1]=107; testIndex[2]=77;

    TensorUtilities tensUtilties;

    for (itMask.GoToBegin(), itTens.GoToBegin(); !itMask.IsAtEnd(), !itTens.IsAtEnd();
    		++itMask, ++itTens)
    {
    	if (itMask.Get() != 0)
    	{
    		DiffusionType D, temp;
    		D= itTens.Get();
    		RealType temp_x;
    		temp = D;
    		temp_x = D[2];
    		D[2] = D[3];
    		D[3] = temp_x;

    		std::cout << "D " << D << std::endl;
    		std::cout << "temp " << temp << std::endl;
    		itTens.Set(temp);

    	}
    }

    typedef itk::ImageFileWriter<TensorImageType>  WriterType;
    WriterType::Pointer writer = WriterType::New();

    writer->SetFileName("Swapped.nii.gz");
    writer->SetInput(tensorImage1);
    writer->Update();

    return 0;
}
