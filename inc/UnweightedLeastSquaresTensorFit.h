/*
 * UnweightedLeastSquaresTensorFit.h
 *
 *  Created on: Jul 24, 2015
 *      Author: vgupta
 */

#ifndef INC_UNWEIGHTEDLEASTSQUARESTENSORFIT_H_
#define INC_UNWEIGHTEDLEASTSQUARESTENSORFIT_H_

#include "TensorUtilites.h"
#include "itkDiffusionTensor3D.h"
#include "itkImage.h"
#include "itkImageRegionIterator.h"
#include "itkVector.h"
#include "itkMatrix.h"
#include "vnl/vnl_matrix.h"
#include "vnl/vnl_inverse.h"


class UnweightedLeastSquaresTensorEstimation
{
	typedef float RealType;
//	const int ImageDim =3;
	typedef itk::DiffusionTensor3D<RealType> DiffusionTensorType;
	typedef itk::Image<DiffusionTensorType, 3> TensorImageType;

	typedef itk::Image<RealType, 3> ScalarImageType;
	typedef std::vector<ScalarImageType::Pointer> ImageListType;

        typedef itk::Vector<double, 3> VectorType;
        typedef itk::Image<VectorType, 3> VectorImageType;
        typedef std::vector<VectorImageType::Pointer> VectorImageListType;


	typedef itk::ImageRegionIterator<ScalarImageType> ScalarImageIterator;
	typedef itk::ImageRegionIterator<VectorImageType> VectorImageIterator;
	typedef itk::ImageRegionIterator<TensorImageType> TensorImageIterator;

public:
	void ReadGradientList(VectorImageListType gradList);
	void ReadDWIList(ImageListType DWIList);
	void ReadMask(ScalarImageType::Pointer maskImage);
	void ReadBVal(RealType bVal);
	void ReadB0Image(ScalarImageType::Pointer B0Image);

	TensorImageType::Pointer Compute();
private:
	void TensorEstimate();
	ImageListType m_DWIList;
	VectorImageListType m_GradList;
	ScalarImageType::Pointer m_maskImage;
	RealType m_BVal;
	ScalarImageType::Pointer m_B0Image;

};



#endif /* INC_UNWEIGHTEDLEASTSQUARESTENSORFIT_H_ */
