/*
 * JointTensorEstimation.h
 *
 *  Created on: Jul 28, 2015
 *      Author: vgupta
 */

#ifndef INC_JOINTTENSORESTIMATION_H_
#define INC_JOINTTENSORESTIMATION_H_


#include "vnl/algo/vnl_svd.h"
#include "vnl/vnl_matrix.h"
#include "itkDiffusionTensor3D.h"
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "vnl/vnl_diag_matrix.h"
#include "itkImageRegionIterator.h"
#include "itkNeighborhoodIterator.h"
#include "TensorUtilites.h"
#include "itkNeighborhoodIterator.h"
#include "itkAddImageFilter.h"
#include "itkMultiplyImageFilter.h"
#include "itkDivideImageFilter.h"
#include "TotalEnergy.h"

class JointTensorEstimation{
	typedef float RealType;
	typedef itk::DiffusionTensor3D<RealType> DiffusionTensorType;
	typedef itk::Image<DiffusionTensorType, 3> TensorImageType;

	typedef itk::Image<RealType, 3> ScalarImageType;
	typedef std::vector<ScalarImageType::Pointer> ImageListType;

    typedef itk::Vector<RealType, 3> VectorType;
    typedef itk::Image<VectorType, 3> VectorImageType;
    typedef std::vector<VectorImageType::Pointer> VectorImageListType;
    typedef vnl_matrix<RealType> MatrixType;

    typedef itk::MultiplyImageFilter<TensorImageType, TensorImageType, TensorImageType> MultiplyByScalarTensorImageFilterType;
    typedef itk::AddImageFilter<TensorImageType, TensorImageType, TensorImageType> AddTensorImageFilterType;

	typedef itk::ImageRegionIterator<ScalarImageType> ScalarImageIterator;
	typedef itk::ImageRegionIterator<VectorImageType> VectorImageIterator;
	typedef itk::ImageRegionIterator<TensorImageType> TensorImageIterator;
	typedef itk::NeighborhoodIterator<TensorImageType> TensorNeighIterator;
	typedef itk::NeighborhoodIterator<ScalarImageType> ScalarNeighIterator;

	typedef itk::ImageFileWriter<ScalarImageType> ScalarWriterType;
	typedef itk::ImageFileWriter<TensorImageType> TensorWriterType;



	typedef itk::DivideImageFilter<ScalarImageType, ScalarImageType, ScalarImageType> DivideImageFilterType;
public:
	void ReadDWIList(ImageListType list);
	void ReadGradientList(VectorImageListType veclist);
	void ReadLamba(RealType lambda);
	void ReadSigma(vnl_vector<RealType> sigma);
	void ReadBVal(RealType Bval);
	void ReadMaskImage(ScalarImageType::Pointer maskImage);
	void ReadInitialTensorImage(TensorImageType::Pointer tensorImage);
	TensorImageType::Pointer ComputeDelSim();
	void UpdateTensors();
	void ReadKappa(RealType kappa);
	void ReadB0Image(ScalarImageType::Pointer B0Image);

	void ReadStepSize(RealType m_StepSize);
	void ReadNumOfIterations(int m_Iterations);
	void ReadB0Thres(RealType thres);


	ScalarImageType::Pointer ComputePsiImage(ScalarImageType::Pointer image); // tested OK
	ScalarImageType::Pointer GradientLogMagTensorImage(TensorImageType::Pointer logTensorImage); //checked tested OK
	TensorImageType::Pointer ComputeLaplaceTensor(TensorImageType::Pointer logTensorImage); //checked tested OK

	TensorImageType::Pointer ComputeFirstTermDelReg(ScalarImageType::Pointer PsiImage, TensorImageType::Pointer LaplaceTensorImage);
	TensorImageType::Pointer ComputeSecondTermDelReg(ScalarImageType::Pointer PsiImage, TensorImageType::Pointer LogTensorImage);
	TensorImageType::Pointer ComputeDelReg(TensorImageType::Pointer FirstTerm, TensorImageType::Pointer SecTerm);

	TensorImageType::Pointer ComputeDelSim_Frac(TensorImageType::Pointer LogtensorImage);
	TensorImageType::Pointer ComputeDelSim_nonFrac(TensorImageType::Pointer LogtensorImage);


	TensorImageType::Pointer UpdateTerms();

private:
	void GaussianNoise();
	ImageListType m_DWIList;
	VectorImageListType m_GradList;
	ScalarImageType::Pointer m_maskImage;
	RealType m_Lambda;
	vnl_vector<RealType> m_Sigma;
	TensorImageType::Pointer m_dt_init;
	RealType m_bval, m_kappa, m_Step_size;
	int m_Iterations;
	ScalarImageType::Pointer m_B0Image;
	RealType m_thres;
};



#endif /* INC_JOINTTENSORESTIMATION_H_ */
