/*
 * CopyImage.h
 *
 *  Created on: Jul 29, 2015
 *      Author: vgupta
 */

#ifndef INC_COPYIMAGE_H_
#define INC_COPYIMAGE_H_

#include "itkImage.h"
#include "itkDiffusionTensor3D.h"


class CopyImage{
	typedef float RealType;
	typedef itk::Image<RealType, 3> ScalarImageType;
	typedef itk::DiffusionTensor3D<RealType> DiffusionTensorType;
	typedef itk::Image<DiffusionTensorType, 3> TensorImageType;

public:
	void CopyScalarImage(ScalarImageType::Pointer srcImage, ScalarImageType::Pointer trgImage );
	void CopyTensorImage(TensorImageType::Pointer srcImage, TensorImageType::Pointer trgImage );
};



#endif /* INC_COPYIMAGE_H_ */
