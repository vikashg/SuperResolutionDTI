/*
 * UnweightedLeastSquaresTensorFit.cxx
 *
 *  Created on: Jul 24, 2015
 *      Author: vgupta
 */
#include "../inc/UnweightedLeastSquaresTensorFit.h"
#include "vnl/algo/vnl_svd.h"
#include "vnl/algo/vnl_matrix_inverse.h"
#include "vnl/algo/vnl_symmetric_eigensystem.h"

void UnweightedLeastSquaresTensorEstimation::ReadDWIList(ImageListType _DWIList)
{
	 m_DWIList = _DWIList;
}

void UnweightedLeastSquaresTensorEstimation::ReadGradientList(VectorImageListType _gradList)
{
	 m_GradList =_gradList;
}

void UnweightedLeastSquaresTensorEstimation::ReadMask(ScalarImageType::Pointer _maskImage)
{
	m_maskImage =_maskImage;
}

void UnweightedLeastSquaresTensorEstimation::ReadBVal(RealType bVal)
{
	m_BVal = bVal;
}

void UnweightedLeastSquaresTensorEstimation::ReadB0Image(ScalarImageType::Pointer image)
{
	m_B0Image = image;
}

UnweightedLeastSquaresTensorEstimation::TensorImageType::Pointer UnweightedLeastSquaresTensorEstimation::
Compute()
{
	TensorImageType::Pointer tensorImage = TensorImageType::New();
	tensorImage->SetSpacing(m_DWIList[0]->GetSpacing());
	tensorImage->SetRegions(m_DWIList[0]->GetLargestPossibleRegion());
	tensorImage->SetOrigin(m_DWIList[0]->GetOrigin());
	tensorImage->SetDirection(m_DWIList[0]->GetDirection());
	tensorImage->Allocate();

	ScalarImageIterator itMask(m_maskImage, m_maskImage->GetLargestPossibleRegion());
	TensorImageIterator itTensor(tensorImage, tensorImage->GetLargestPossibleRegion());

	const int numOfGradients = m_GradList.size();

	vnl_matrix<float> Y;
	Y.set_size(numOfGradients,1);

	for (itMask.GoToBegin(), itTensor.GoToBegin(); !itMask.IsAtEnd(), !itTensor.IsAtEnd(); ++itMask, ++itTensor)
	{
		if (itMask.Get() != 0)
		{
			vnl_matrix<RealType> H;
			H.set_size(numOfGradients,6);

			RealType S_0 = m_B0Image->GetPixel(itMask.GetIndex());

			for (int i=0; i < m_GradList.size(); i++)
			{
			 VectorType G;
			 G=m_GradList[i]->GetPixel(itMask.GetIndex());
			 RealType gx, gy, gz;
			 gx=G[0]; gy=G[1]; gz=G[2];
			 itk::Vector<RealType, 6>  H_i;

			 H_i.SetElement(0, gx*gx);
			 H_i.SetElement(1, gy*gy);
			 H_i.SetElement(2, gz*gz);
			 H_i.SetElement(3, 2*gx*gy);
			 H_i.SetElement(4, 2*gx*gz);
			 H_i.SetElement(5, 2*gy*gz);

			 H.set_row(i, H_i.GetVnlVector());

			 //Make the Diffusion Image vector Y
			 RealType S_i = m_DWIList[i]->GetPixel(itMask.GetIndex());
			 Y.set_row(i, log(S_0/S_i)/m_BVal);

			}
			//Make the pseudoInverse
			vnl_matrix<float> H_square = H.transpose()*H;
     		vnl_matrix<RealType> D_vec = vnl_matrix_inverse<RealType>(H_square)*H.transpose()*Y;

			itk::DiffusionTensor3D<RealType> D;

			D(0,0) = D_vec.get(0,0);
			D(1,1) = D_vec.get(1,0);
			D(2,2) = D_vec.get(2,0);
			D(0,1) = D_vec.get(3,0);
			D(0,2) = D_vec.get(4,0);
			D(1,2) = D_vec.get(5,0);


//			std::cout << D << std::endl;

			vnl_matrix<RealType> D_mat;
			D_mat.set_size(3,3);

			for (int i=0; i < 3; i++)
			{
				for (int j=0; j < 3 ; j++)
				{
					D_mat(i,j) = D(i,j);
				}
			}

			if ((D_mat.has_nans() == 0) && (D_mat.is_finite() == 1))
			{
			itTensor.Set(D);

//			std::cout << D << std::endl;
//			TensorUtilities tensUtils;
//			tensUtils.LogM(D);
			}

		}
	}


	return tensorImage;
}


