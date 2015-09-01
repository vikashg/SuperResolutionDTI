/*
 * JacobianComputation.h
 *
 *  Created on: Aug 24, 2015
 *      Author: vgupta
 */

#ifndef INC_JACOBIANCOMPUTATION_H_
#define INC_JACOBIANCOMPUTATION_H_

#include "itkImage.h"
#include "itkVector.h"
#include "itkImageRegionIterator.h"
#include "itkNeighborhoodIterator.h"
#include "itkImageFileWriter.h"

class JacobianComputation{
	typedef float RealType;
	typedef itk::Image<RealType, 3> ImageType;
	typedef itk::Vector<RealType, 3> VectorType;
	typedef itk::Image<VectorType, 3>  VectorImageType;
	typedef itk::Matrix<float, 3, 3> MatrixType;
	typedef itk::Image<MatrixType,3> MatrixImageType;

	typedef std::vector<VectorType> GradientListType;

public:
	void ReadDefField(VectorImageType::Pointer defField);
	void ReadMaskImage(ImageType::Pointer maskImage);
	void ReadGradientList(GradientListType gradList);
	ImageType::Pointer Compute();

private:
	ImageType::Pointer m_MaskImage;
	VectorImageType::Pointer m_DefField;
	GradientListType m_GradList;
};



#endif /* INC_JACOBIANCOMPUTATION_H_ */
