/*
 * ComposeImage.cpp
 *
 *  Created on: Jul 4, 2015
 *      Author: vgupta
 */

#include "ComposeImage.h"

ComposeImageFilter::ImageType::Pointer ComposeImageFilter::GetLRImage(ImageType::Pointer image)
{
	m_imageLR = image;
	std::cout << "Read LR image" << std::endl;
}

ComposeImageFilter::ImageType::Pointer ComposeImageFilter::GetHRImage(ImageType::Pointer image)
{

	m_imageHR = image;
	std::cout << "Read HR image" << std::endl;
}

void ComposeImageFilter::ReadMatrix(SparseMatrixType matrix)
{
	m_matrix = matrix;
	std::cout << "Read Matrix" << std::endl;
}

ComposeImageFilter::ImageType::IndexType ComposeImageFilter::ComputeImageIndex(long int N, ImageType::Pointer image)
{
	ImageType::IndexType index;
	ImageType::SizeType size = image->GetLargestPossibleRegion().GetSize();

	    std::div_t div_1, div_2;
	    long int temp1=size[0]*size[1];

	    div_1 = div(N,temp1);
	    index[2]=div_1.quot;

	    long int temp2=size[0];
	    div_2 = div(div_1.rem, temp2);
	    index[1]=div_2.quot;
	    index[0]=div_2.rem;
	    return index;

}

ComposeImageFilter::ImageType::Pointer ComposeImageFilter::ComposeIt()
{
	ImageType::Pointer outputImage = ImageType::New();

	outputImage->SetRegions(m_imageLR->GetLargestPossibleRegion());
	outputImage->SetOrigin(m_imageLR->GetOrigin());
	outputImage->SetSpacing(m_imageLR->GetSpacing());
	outputImage->SetDirection(m_imageLR->GetDirection());
	outputImage->Allocate();
	outputImage->FillBuffer(0.0);

	std::cout << "outputImage" << std::endl;

	ImageType::IndexType testIndex;
	testIndex[0] = 0; testIndex[1]=0; testIndex[2]=0;

	std::cout << m_matrix.rows() << " " << m_matrix.cols() << std::endl;

	for (int i=0; i< m_matrix.rows() ; i++ )
	{
		vcl_vector<int> rowIndices;
		vcl_vector<float> rowValues;
	    vcl_vector<pair_t> rowM = m_matrix.get_row(i);

		float sum_vox =0;

		ImageType::IndexType outputIndex;
		outputIndex = ComputeImageIndex(i, m_imageLR);
		
//		std::cout << "ImageLR " << " " << i << " " <<  outputIndex << std::endl;

		for (vcl_vector<pair_t>::const_iterator it = rowM.begin(); it != rowM.end(); ++it)
		{
			ImageType::IndexType HRIndex;
			HRIndex = ComputeImageIndex(it->first, m_imageHR);
			sum_vox += (it->second)* (m_imageHR->GetPixel(HRIndex));
//			std::cout << HRIndex <<  " " << it->second <<  std::endl;

		}
//		std::cout << sum_vox << std::endl;
		outputImage->SetPixel(outputIndex, sum_vox);
//


	}

	return outputImage;
}

