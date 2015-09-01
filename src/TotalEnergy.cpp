/*
 * TotalEnergy.cxx
 *
 *  Created on: Aug 11, 2015
 *      Author: vgupta
 */

#include "TotalEnergy.h"

void TotalEnergy::ReadGradientImageList(VectorImageListType GradList)
{
	m_GradList = GradList;
}

void TotalEnergy::ReadMaskImage(ScalarImageType::Pointer image)
{
	m_MaskImage = image;
}

void TotalEnergy::ReadB0Image(ScalarImageType::Pointer image)
{
	m_B0Image = image;
}

void TotalEnergy::ReadImageList(ImageListType DWIList)
{
	m_DWIList = DWIList;
}

void TotalEnergy::ReadBValue(RealType bVal)
{
	m_BVal = bVal;
}

void TotalEnergy::ReadKappa(RealType kappa)
{
	m_Kappa = kappa;
}

void TotalEnergy::ReadSigma(vnl_vector<RealType> sigma)
{
	m_Sigma = sigma;
}

TotalEnergy::RealType TotalEnergy::GaussianNoise_Frac(TensorImageType::Pointer LogtensorImage)
{
	RealType Tots=0;

		ScalarImageIterator itMask(m_MaskImage, m_MaskImage->GetLargestPossibleRegion());
		TensorImageIterator itLogTens(LogtensorImage, LogtensorImage->GetLargestPossibleRegion());
		ScalarImageIterator itB0(m_B0Image, m_B0Image->GetLargestPossibleRegion());

		int numOfImages = m_DWIList.size();
		TensorUtilities utilsTensor;

		ScalarImageType::IndexType testIndex;
		testIndex[0]=23; testIndex[1]=39; testIndex[2]=7;

		for (itLogTens.GoToBegin(), itB0.GoToBegin(), itMask.GoToBegin();
			!itLogTens.IsAtEnd(), !itB0.IsAtEnd(), !itMask.IsAtEnd();
			++itLogTens, ++itB0, ++itMask)
		{
			if (itMask.Get() != 0)
			{
				for (int i=0; i < numOfImages; i++)
				{
				RealType atten_i;
				vnl_vector<RealType> g_i = m_GradList[i]->GetPixel(itMask.GetIndex()).GetVnlVector();
				vnl_matrix<RealType> g_mat_i;
				g_mat_i.set_size(3,1);
				g_mat_i.set_column(0,g_i);


				DiffusionTensorType D = utilsTensor.ExpM(itLogTens.Get());
				MatrixType D_mat;
				D_mat.set_size(3,3);
				D_mat = utilsTensor.ConvertDT2Mat(D);

				MatrixType temp; temp.set_size(1,1);
				temp = g_mat_i.transpose()*D_mat*g_mat_i;

				atten_i = exp(temp(0,0)*(-1)*m_BVal);
				RealType obs_i = m_DWIList[i]->GetPixel(itMask.GetIndex())/itB0.Get();

				RealType error_noise = (obs_i - atten_i)*(obs_i - atten_i)/(m_Sigma[i]*m_Sigma[i]);

				 TensorImageType::IndexType testIndex;
				 testIndex[0]=54; testIndex[1]=21; testIndex[2]=1;

	//			if (itMask.GetIndex() == testIndex)
	//			{
	//			std::cout << "In the Total Energy " << itLogTens.Get() << std::endl;
	//			}

	//			if (isnan(error_noise) == 1)
	//			{
	//				std::cout << "error_noise " << itMask.GetIndex() << " " << itLogTens.Get() <<  std::endl;
	//			}
				Tots = Tots + error_noise;

				}
			}
		}

		return Tots;
}

TotalEnergy::RealType TotalEnergy::GaussianNoise(TensorImageType::Pointer LogtensorImage)
{
	RealType Tots=0;

	ScalarImageIterator itMask(m_MaskImage, m_MaskImage->GetLargestPossibleRegion());
	TensorImageIterator itLogTens(LogtensorImage, LogtensorImage->GetLargestPossibleRegion());
	ScalarImageIterator itB0(m_B0Image, m_B0Image->GetLargestPossibleRegion());

	int numOfImages = m_DWIList.size();
	TensorUtilities utilsTensor;

	ScalarImageType::IndexType testIndex;
	testIndex[0]=23; testIndex[1]=39; testIndex[2]=7;

	for (itLogTens.GoToBegin(), itB0.GoToBegin(), itMask.GoToBegin();
		!itLogTens.IsAtEnd(), !itB0.IsAtEnd(), !itMask.IsAtEnd();
		++itLogTens, ++itB0, ++itMask)
	{
		if (itMask.Get() != 0)
		{
			for (int i=0; i < numOfImages; i++)
			{
			RealType atten_i;
			vnl_vector<RealType> g_i = m_GradList[i]->GetPixel(itMask.GetIndex()).GetVnlVector();
			vnl_matrix<RealType> g_mat_i;
			g_mat_i.set_size(3,1);
			g_mat_i.set_column(0,g_i);


			DiffusionTensorType D = utilsTensor.ExpM(itLogTens.Get());
			MatrixType D_mat;
			D_mat.set_size(3,3);
			D_mat = utilsTensor.ConvertDT2Mat(D);

			MatrixType temp; temp.set_size(1,1);
			temp = g_mat_i.transpose()*D_mat*g_mat_i;

			atten_i = exp(temp(0,0)*(-1)*m_BVal)*itB0.Get();
			RealType obs_i = m_DWIList[i]->GetPixel(itMask.GetIndex());

			RealType error_noise = (obs_i - atten_i)*(obs_i - atten_i)/(m_Sigma[i]*m_Sigma[i]);

			 TensorImageType::IndexType testIndex;
			 testIndex[0]=54; testIndex[1]=21; testIndex[2]=1;

//			if (itMask.GetIndex() == testIndex)
//			{
//			std::cout << "In the Total Energy " << itLogTens.Get() << std::endl;
//			}

//			if (isnan(error_noise) == 1)
//			{
//				std::cout << "error_noise " << itMask.GetIndex() << " " << itLogTens.Get() <<  std::endl;
//			}
			Tots = Tots + error_noise;

			}
		}
	}

	return Tots;
}

TotalEnergy::RealType TotalEnergy::RegularizationEnergy(TensorImageType::Pointer logTensorImage)
{
	JointTensorEstimation jTestimation;
	jTestimation.ReadMaskImage(m_MaskImage);
	ScalarImageType::Pointer gradMagTensorImage = jTestimation.GradientLogMagTensorImage(logTensorImage) ;

	ScalarImageIterator itMask(m_MaskImage, m_MaskImage->GetLargestPossibleRegion());
	ScalarImageIterator itGrad(gradMagTensorImage, gradMagTensorImage->GetLargestPossibleRegion());

	RealType Total =0;
	for (itMask.GoToBegin(), itGrad.GoToBegin(); !itMask.IsAtEnd(), !itGrad.IsAtEnd();
			++itMask, ++itGrad)
	{
		RealType regTerm, temp;
		temp = itGrad.Get()/m_Kappa;
		temp = 2*sqrt(1+ temp*temp) -2;
		Total = Total + temp;
	}

	return Total;
}

