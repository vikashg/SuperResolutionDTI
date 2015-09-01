/*
 * MapFIlterLR2HRDispField.h
 *
 *  Created on: Aug 30, 2015
 *      Author: vgupta
 */
#define TOTAL_PTS 9000000

#ifndef INC_MAPFILTERLR2HRDISPFIELD_H_
#define INC_MAPFILTERLR2HRDISPFIELD_H_

#include "itkImage.h"
#include "itkPointSet.h"
#include "itkPoint.h"
#include <iostream>
#include <vnl/vnl_sparse_matrix.h>
#include "itkImageMaskSpatialObject.h"
#include <vcl_vector.h>
#include "itkTransform.h"
#include "itkAffineTransform.h"
#include "itkTransformFileReader.h"
#include "itkVector.h"
#include "itkDisplacementFieldTransform.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

class MapFilterLR2HR1
{
	typedef float RealType;
	typedef itk::Image<RealType,3> ImageType;
	typedef itk::Point<RealType, 3> PointType;
//	typedef PointSetType::PointType PointType;
//	typedef PointSetType::PointsContainerPointer PointsContainerPointer;
	typedef itk::TransformFileReader TransformFileReaderType;
	typedef TransformFileReaderType::TransformListType TransformListType;
	typedef itk::TransformBase TransformBaseType;
	typedef itk::AffineTransform<RealType, 3> AffineTransformType;

    typedef itk::ImageMaskSpatialObject<3> MaskSpatialObjectType;
    typedef MaskSpatialObjectType::ImageType MaskSpatialImageType;
    typedef itk::ImageFileReader<MaskSpatialImageType> MaskSpatialImageReader;


	typedef itk::Vector<RealType,3> VectorType;
	typedef itk::Image<VectorType,3> DisplacementFieldImageType;
	typedef itk::DisplacementFieldTransform<RealType, 3> DisplacementFieldTransformType;
public:
	void ComputeMapWithDefField( );
	vnl_sparse_matrix<float> GetLR2HRMatrix();
	vnl_sparse_matrix<float> GetHR2LRMatrix();

	void ReadFixedImage(ImageType::Pointer imageLR);
	void ReadMovingImage(ImageType::Pointer imageHR);
	void ReadMaskImage(MaskSpatialImageType::Pointer image);
	void ReadDeformationField(DisplacementFieldImageType::Pointer image);

private:
	unsigned long int ComputeMatrixIndex(ImageType::Pointer image, ImageType::IndexType index);
	ImageType::PointType m_Origin_FI, m_Origin_MI, m_Final_FI, m_Final_MI;
	ImageType::SpacingType m_spacing_FI, m_spacing_MI;

	vnl_sparse_matrix<float> m_SpVnl_Row_normalized, m_SpVnl_Col_normalized; //
	AffineTransformType::Pointer m_AffineTransform;
	ImageType::Pointer m_fixedImage;
	ImageType::Pointer m_MovingImage;
	MaskSpatialImageType::Pointer m_MaskImage;
	DisplacementFieldImageType::Pointer m_DispField;

};








#endif /* INC_MAPFILTERLR2HRDISPFIELD_H_ */
