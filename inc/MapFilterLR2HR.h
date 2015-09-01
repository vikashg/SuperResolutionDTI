/*
 * MapFilterLR2HR.h
 *
 *  Created on: Jul 3, 2015
 *      Author: vgupta
 */

#ifndef INC_MAPFILTERLR2HR_H_
#define INC_MAPFILTERLR2HR_H_

#include "itkImage.h"
#include "itkPointSet.h"
#include "itkPoint.h"
#include <iostream>
#include <vnl/vnl_sparse_matrix.h>
#include <vcl_vector.h>
#include "itkTransform.h"
#include "itkAffineTransform.h"
#include "itkTransformFileReader.h"
#include "itkVector.h"
#include "itkInverseDisplacementFieldImageFilter.h"
#include "itkDisplacementFieldTransform.h"

class MapFilterLR2HR1
{
	typedef itk::Image<float,3> ImageType;
	typedef itk::PointSet<double, 3> PointSetType;
	typedef PointSetType::PointType PointType;
	typedef PointSetType::PointsContainerPointer PointsContainerPointer;
	typedef itk::TransformFileReader TransformFileReaderType;
	typedef TransformFileReaderType::TransformListType TransformListType;
	typedef itk::TransformBase TransformBaseType;
	typedef itk::AffineTransform<double, 3> AffineTransformType;

	typedef itk::Vector<float,3> VectorType;
	typedef itk::Image<VectorType,3> DisplacementFieldImageType;
	typedef itk::DisplacementFieldTransform<float, 3> DisplacementFieldTransformType;
public:
	void ComputeMap( );
	void ComputeMapWithDefField( );
	vnl_sparse_matrix<float> GetLR2HRMatrix();
	vnl_sparse_matrix<float> GetHR2LRMatrix();

	void ReadLRImage(ImageType::Pointer imageLR);
	void ReadHRImage(ImageType::Pointer imageHR);
	void ReadAffineTransform(AffineTransformType::Pointer affineTransform);
	void ReadMaskImage(ImageType::Pointer image);
	void ReadDeformationField(DisplacementFieldImageType::Pointer dispField);

private:
	unsigned long int ComputeMatrixIndex1(ImageType::Pointer image, ImageType::IndexType index);
	ImageType::PointType m_Origin_LR, m_Origin_HR, m_Final_LR, m_Final_HR;
	ImageType::SpacingType m_spacing_LR, m_spacing_HR;

	vnl_sparse_matrix<float> m_SpVnl_Row_normalized, m_SpVnl_Col_normalized;
	AffineTransformType::Pointer m_AffineTransform;
	ImageType::Pointer m_imageLR;
	ImageType::Pointer m_imageHR;
	ImageType::Pointer m_MaskImage;
		DisplacementFieldImageType::Pointer m_DispField;

};




#endif /* INC_MAPFILTERLR2HR_H_ */
