/*
 * CopyImage.cxx
 *
 *  Created on: Jul 29, 2015
 *      Author: vgupta
 */


#include "CopyImage.h"

void CopyImage::CopyScalarImage(ScalarImageType::Pointer srcImage, ScalarImageType::Pointer trgImage)
{
	trgImage->SetOrigin(srcImage->GetOrigin());
	trgImage->SetDirection(srcImage->GetDirection());
	trgImage->SetSpacing(srcImage->GetSpacing());
	trgImage->SetRegions(srcImage->GetLargestPossibleRegion());
	trgImage->Allocate();
	trgImage->FillBuffer(0.0);
}

void CopyImage::CopyTensorImage(TensorImageType::Pointer srcImage, TensorImageType::Pointer trgImage)
{
	DiffusionTensorType ZeroD;
	ZeroD.Fill(0.0);
	trgImage->SetOrigin(srcImage->GetOrigin());
	trgImage->SetDirection(srcImage->GetDirection());
	trgImage->SetSpacing(srcImage->GetSpacing());
	trgImage->SetRegions(srcImage->GetLargestPossibleRegion());
	trgImage->Allocate();
	trgImage->FillBuffer(ZeroD);


}
