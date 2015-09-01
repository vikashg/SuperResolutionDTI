#ifndef __ComposeImage_txx
#define __ComposeImage_txx

#include "itkComposeImage.h"
//#include <cmath>

namespace itk
{
template<class TImage>
ComposeImageFilter<TImage>::ComposeImageFilter()
{
    this->SetNumberOfInputs(2);
}

template<class TImage>
void ComposeImageFilter<TImage>::GenerateInputRequestedRegion()
{
    this->Superclass::Superclass::GenerateInputRequestedRegion();
}


template<class TImage>
void ComposeImageFilter<TImage>::SetSourceImage(const TImage* imageSrc)
{
    this->SetNthInput(0, const_cast<TImage*> (imageSrc));
}

template<class TImage>
void ComposeImageFilter<TImage>::SetTargetImage(const TImage* imageTrg)
{
    this->SetNthInput(1, const_cast<TImage*> (imageTrg));
}

template<class TImage>
typename TImage::ConstPointer ComposeImageFilter<TImage>::GetSourceImage()
{
    return static_cast<const TImage *>
            (this->ProcessObject::GetInput(0));
}

template<class TImage>
typename TImage::ConstPointer ComposeImageFilter<TImage>::GetTargetImage()
{
    return static_cast<const TImage *>
            (this->ProcessObject::GetInput(1));
}

template<class TImage>
void ComposeImageFilter<TImage>::ReadMapMatrix(vnl_sparse_matrix<float> matrix)
{
    m_matrix = matrix;
}

template<class TImage>
typename TImage::IndexType ComposeImageFilter<TImage>::ComputeImageIndex(long int N, typename TImage::ConstPointer image)
{
    typename TImage::SizeType size;
    size = image->GetLargestPossibleRegion().GetSize();

    std::div_t div_1, div_2;
    long int temp1=size[0]*size[1];

    typename TImage::IndexType index;
    div_1 = div(N,temp1);
    index[2]=div_1.quot;

    long int temp2=size[0];
    div_2 = div(div_1.rem, temp2);
    index[1]=div_2.quot;
    index[0]=div_2.rem;
    return index;

}

template<class TImage>
void ComposeImageFilter<TImage>
::GenerateData()
{
    typename TImage::ConstPointer srcImage = this->GetSourceImage();
    typename TImage::ConstPointer trgImage = this->GetTargetImage();

    // Setup Output Image
    typename TImage::Pointer output = this->GetOutput();
    output->SetRegions(srcImage->GetLargestPossibleRegion());
    output->SetSpacing(srcImage->GetSpacing());
    output->Allocate();
    output->FillBuffer(0);

//    std::cout << m_matrix.rows() << std::endl;

    typename TImage::IndexType testIndex; testIndex.Fill(14) ;
//    testIndex[0] = 97; testIndex[1] = 53; testIndex[2]=8;
//    testIndex[0] = 35; testIndex[1] = 47; testIndex[2]=37;

    for (int i=0; i<m_matrix.rows(); i++)
    {
      vcl_vector<pair_t> rowM = m_matrix.get_row(i);
      vcl_vector<int> rowIndices;
      vcl_vector<float> rowValues;

      float sum_vox =0;

      typename TImage::IndexType outputIndex;
      outputIndex = ComputeImageIndex(i, srcImage);


//      std::cout << "Src Index: " << outputIndex << std::endl;

      for (vcl_vector<pair_t>::const_iterator it = rowM.begin(); it != rowM.end(); ++it)
      {
                    typename TImage::IndexType trgIndex;
                    trgIndex = ComputeImageIndex(it->first, trgImage);
                    sum_vox += (it->second)*trgImage->GetPixel(trgIndex);

//                    if (outputIndex == testIndex)
//                    {
//                        std::cout << " Compose Image " << trgIndex << " " << it->second << " " << trgImage->GetPixel(trgIndex) << " " << (it->second)*trgImage->GetPixel(trgIndex) << std::endl;
//                        std::cout << "SumVox " << sum_vox << std::endl;
//                    }


     }

//      if (outputIndex == tempIndex)
//      std::cout << sum_vox << std::endl;
      output->SetPixel(outputIndex, sum_vox);

    }

}

template<class TImage>
DataObject::Pointer ComposeImageFilter<TImage>::MakeOutput(unsigned int idx)
{
    DataObject::Pointer output;

    output = (TImage::New()).GetPointer();
    return output.GetPointer();

}

template<class TImage>
TImage* ComposeImageFilter<TImage>::GetOutput()
{
    return dynamic_cast<TImage *>(this->ProcessObject::GetOutput(0));

}

}

#endif
