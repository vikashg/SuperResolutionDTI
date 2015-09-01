/*
 * ExtractTensorImageRegion.cxx
 *
 *  Created on: Aug 1, 2015
 *      Author: vgupta
 */


#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkExtractImageFilter.h"
#include "../GetPot/Getpot"

#include "itkDiffusionTensor3D.h"

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

    const string image_n =cl.follow("NoFile", 1, "-i");
    const string out_n = cl.follow("NoFile", 1, "-o");

    const int idx = cl.follow(0, 1, "-ix");
    const int idy = cl.follow(0, 1, "-iy");
    const int idz = cl.follow(0, 1, "-iz");

    const int sx = cl.follow(0, 1, "-sx");
    const int sy = cl.follow(0, 1, "-sy");
    const int sz = cl.follow(0, 1, "-sz");

    typedef itk::DiffusionTensor3D<float> DiffusionTensorType;

    typedef itk::Image< DiffusionTensorType, 3> ImageType;
    typedef itk::ImageFileReader<ImageType> ReaderType;
    ReaderType::Pointer reader = ReaderType::New();

    reader->SetFileName(image_n);
    reader->Update();

    ImageType::Pointer image = reader->GetOutput();

    ImageType::SizeType size;
    ImageType::IndexType id;

    id[0] = idx; id[1] = idy; id[2] = idz;

    size[0] = sx; size[1] = sy; size[2] =sz;

    ImageType::RegionType region(id, size);

    typedef itk::ExtractImageFilter<ImageType, ImageType> FilterType;
    FilterType::Pointer filter = FilterType::New();

    filter->SetExtractionRegion(region);
    filter->SetInput(image);
    filter->SetDirectionCollapseToIdentity();

    filter->Update();

    ImageType::Pointer exImage = filter ->GetOutput();

    typedef itk::ImageFileWriter<ImageType> WriterType;
    WriterType::Pointer writer = WriterType::New();

    writer->SetFileName(out_n);
    writer->SetInput(exImage);
    writer->Update();

    return 0;
}
