#ifndef __MAPLR2HR_txx
#define __MAPLR2HR_txx

#include "itkMapLR2HR.h"
#include "sobol.hpp"

namespace itk
{
template<class TImage>
MapLR2HRFilter<TImage>::MapLR2HRFilter()
{
	this->SetNumberOfRequiredInputs(2);
}

template<class TImage>
void MapLR2HRFilter<TImage>::GenerateInputRequestedRegion()
{
	this->Superclass::Superclass::GenerateInputRequestedRegion();
}

template<class TImage>
void MapLR2HRFilter<TImage>::SetLRImage(const TImage* imageLR)
{
	this->SetNthInput(0, const_cast<TImage*> (imageLR));
}

template<class TImage>
void MapLR2HRFilter<TImage>::SetHRImage(const TImage* imageHR)
{
	this->SetNthInput(1, const_cast<TImage*> (imageHR));
}

template<class TImage>
void MapLR2HRFilter<TImage>::SetTransform(AffineTransformType::Pointer transform)
{
	this->m_Transform = transform;
	std::cout << transform->GetParameters() << std::endl;
}

template<class TImage>
typename TImage::ConstPointer MapLR2HRFilter<TImage>::GetLRImage()
{
	return static_cast<const TImage *>
	( this->ProcessObject::GetInput(0) );
}

template<class TImage>
typename TImage::ConstPointer MapLR2HRFilter<TImage>::GetHRImage()
{
	return static_cast<const TImage *>
	( this->ProcessObject::GetInput(1) );
}

template<class TImage>
unsigned long int MapLR2HRFilter<TImage>::ComputeMatrixIndex(typename TImage::IndexType index, typename TImage::ConstPointer image)
{
	typename TImage::SizeType size;
	size = image->GetLargestPossibleRegion().GetSize();
	unsigned long int N = index[2]*size[0]*size[1] + index[1]*size[1] + index[0];
	return N;
}


template<class TImage>
void MapLR2HRFilter<TImage>::GenerateData()
{
	std::cout << "Running generate Data " << std::endl;
}


}

#endif
