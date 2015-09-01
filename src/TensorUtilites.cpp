/*
= * TensorUtilites.cxx
 *
 *  Created on: Jul 27, 2015
 *      Author: vgupta
 */

#include "TensorUtilites.h"

TensorUtilities::TensorImageType::Pointer TensorUtilities::LogTensorImageFilter(TensorImageType::Pointer tensorImage, ScalarImageType::Pointer maskImage)
{
	TensorImageType::Pointer LogTensorImage = TensorImageType::New();
	CopyImage cpImage;
	cpImage.CopyTensorImage(tensorImage, LogTensorImage);

	ScalarIterator itMask(maskImage, maskImage->GetLargestPossibleRegion());
	TensorIterator itDT(tensorImage, tensorImage->GetLargestPossibleRegion());
	TensorIterator itLDT(LogTensorImage, LogTensorImage->GetLargestPossibleRegion());

	TensorUtilities utilsTensor;

	std::cout << "LogTensImageFilter " << std::endl;

	for (itMask.GoToBegin(), itDT.GoToBegin(), itLDT.GoToBegin(); !itMask.IsAtEnd(), !itDT.IsAtEnd(), !itLDT.IsAtEnd();
			++itMask, ++itDT, ++itLDT)
	{
		if (itMask.Get() != 0)
		{

			DiffusionTensorType LogDT;
			LogDT = utilsTensor.LogM(itDT.Get());
			itLDT.Set(LogDT);
		}
	}

	return LogTensorImage;
}

TensorUtilities::RealType TensorUtilities::ComputeTensorNorm(DiffusionTensorType D)
{
	RealType Norm;

	for (int i=0; i < 6; i++)
	{
		Norm = Norm + D[i]*D[i];
	}

	return sqrt(2*Norm);
}

TensorUtilities::TensorImageType::Pointer TensorUtilities::ExpTensorImageFilter(TensorImageType::Pointer tensorImage,
																				ScalarImageType::Pointer maskImage)
{
	TensorImageType::Pointer ExpTensorImage = TensorImageType::New();
	CopyImage cpImage;
	cpImage.CopyTensorImage(tensorImage, ExpTensorImage);

	ScalarIterator itMask(maskImage, maskImage->GetLargestPossibleRegion());
	TensorIterator itDT(tensorImage, tensorImage->GetLargestPossibleRegion());
	TensorIterator itEDT(ExpTensorImage, ExpTensorImage->GetLargestPossibleRegion());

	TensorUtilities utilsTensor;

	for (itMask.GoToBegin(), itDT.GoToBegin(), itEDT.GoToBegin() ;!itMask.IsAtEnd(), !itEDT.IsAtEnd(), !itDT.IsAtEnd(); ++itMask, ++itDT, ++itEDT)
	{

					DiffusionTensorType D = ExpM(itDT.Get());
					itEDT.Set(D);

	}

	return ExpTensorImage;

}

TensorUtilities::DiffusionTensorType TensorUtilities::LogM(DiffusionTensorType D)
{
	vnl_matrix<RealType> D_mat, D_log_mat;
	D_mat.set_size(3,3);
	D_log_mat.set_size(3,3);


	D_mat = ConvertDT2Mat(D);
	vnl_symmetric_eigensystem<RealType> eig(D_mat);

	MatrixType R; R.set_size(3,3);
	vnl_vector<RealType> S,L  ; S.set_size(3); L.set_size(3);

	for (int i=0; i < 3; i++)
	{
		S[i] = eig.get_eigenvalue(i);
		R.set_column(i, eig.get_eigenvector(i));
	}

	vnl_diag_matrix<RealType> S_diag, S_log_diag;
	S_diag.set_diagonal(S);


	for (int i=0; i<3; i++)
	{
		L(i) = log(S(i));
	}

	S_log_diag.set_diagonal(L);

	D_log_mat = R*S_log_diag*R.transpose();

	DiffusionTensorType temp;

	temp = ConvertMat2DT(D_log_mat);

	return temp;
}


TensorUtilities::DiffusionTensorType TensorUtilities::ExpM(DiffusionTensorType D)
{

	vnl_matrix<RealType> D_mat, D_exp_mat;
		D_mat.set_size(3,3);
		D_exp_mat.set_size(3,3);

		D_mat = ConvertDT2Mat(D);
		vnl_symmetric_eigensystem<RealType> eig(D_mat);

		MatrixType R; R.set_size(3,3);
		vnl_vector<RealType> S,E  ; S.set_size(3); E.set_size(3);

		for (int i=0; i < 3; i++)
		{
			S[i] = eig.get_eigenvalue(i);
			R.set_column(i, eig.get_eigenvector(i));
		}

		vnl_diag_matrix<RealType> S_diag, S_exp_diag;
		S_diag.set_diagonal(S);

		for (int i=0; i<3; i++)
		{
			E(i) = exp(S(i));
		}

		S_exp_diag.set_diagonal(E);

		D_exp_mat = R*S_exp_diag*R.transpose();

		DiffusionTensorType temp;

		temp = ConvertMat2DT(D_exp_mat);

		return temp;
}

TensorUtilities::MatrixType TensorUtilities::ConvertDT2Mat(DiffusionTensorType D)
{
	MatrixType D_mat; D_mat.set_size(3,3);
	for (int i=0; i < 3; i++)
	{
		for (int j=0; j < 3; j++)
		{
			D_mat(i,j) = D(i,j);
		}
	}

	return D_mat;
}


TensorUtilities::DiffusionTensorType TensorUtilities::ConvertMat2DT(MatrixType D_mat)
{
	DiffusionTensorType D;

	for (int i=0; i < 3 ; i++)
	{
		for (int j=0; j<3; j++)
		{
			D(i,j) = D_mat(i,j);
		}
	}

	return D;
}



TensorUtilities::DiffusionTensorType TensorUtilities::MatrixExpDirDerivative(DiffusionTensorType L, VectorType G, ScalarImageType::IndexType testIndex)
{
	MatrixType L_mat = ConvertDT2Mat(L);


	ScalarImageType::IndexType testIdx;
	testIdx[0]=6; testIdx[1]=49; testIdx[2]=0;

	MatrixType R; R.set_size(3,3);
	vnl_vector<RealType> S  ; S.set_size(3);

	RealType EPS1 =0.00000001;

	vnl_symmetric_eigensystem<RealType> eig(L_mat);

	for (int i=0; i < 3; i++)
	{
		S[i] = eig.get_eigenvalue(i);
		R.set_column(i, eig.get_eigenvector(i));
	}


	vnl_vector<RealType> g = G.GetVnlVector();
	MatrixType G_mat; G_mat.set_size(3,3);


	for (int i=0; i< 3; i++)
	{
		for (int j=0; j<3; j++)
		{
			G_mat(i,j) =  g.get(i)*g.get(j);
		}
	}

	MatrixType RGRt, M;
	RGRt.set_size(3,3); M.set_size(3,3);
	M.fill(0);
	RGRt = R.transpose()*G_mat*R;

	for (int l=0; l < 3 ; l++)
	{
		for (int m=0; m< 3; m++)
		{
			RealType sl,sm;
			sl= S.get(l); sm= S.get(m);

			RealType diff;
			diff = sl - sm;

			if (sl == sm)
			{
				M(l,m) = RGRt(l,m)*exp(sl);

			}
			else
			{
				M(l,m) = RGRt(l,m)*(exp(sm)-exp(sl))/(sm-sl);
			}
		}
	}

	MatrixType Result;
	Result = R.transpose()*M*R;
	DiffusionTensorType  Result_D = ConvertMat2DT(Result);
	return Result_D;

}


TensorUtilities::TensorImageType::Pointer TensorUtilities::ReplaceNaNsReverseEigenValue(TensorImageType::Pointer tensorImage, ScalarImageType::Pointer maskImage)
{
	TensorImageType::Pointer newTensorImage = TensorImageType::New();

		std::cout << "Replace Nans " << std::endl;

		vnl_matrix<RealType> ZeroD_mat; ZeroD_mat.set_size(3,3);
		ZeroD_mat.set_identity();
		DiffusionTensorType ZeroD;

		DiffusionTensorType D_identity = ConvertMat2DT(ZeroD_mat);

		newTensorImage->SetDirection(tensorImage->GetDirection());
		newTensorImage->SetSpacing(tensorImage->GetSpacing());
		newTensorImage->SetOrigin(tensorImage->GetOrigin());
		newTensorImage->SetRegions(tensorImage->GetLargestPossibleRegion());
		newTensorImage->Allocate();
		newTensorImage->FillBuffer(ZeroD);


		TensorIterator itTens1(tensorImage, tensorImage->GetLargestPossibleRegion());
		TensorIterator itTens2(newTensorImage, newTensorImage->GetLargestPossibleRegion());

		ScalarIterator itMask(maskImage, maskImage->GetLargestPossibleRegion());
		std::vector<TensorImageType::IndexType> Index_Matrix;
		int count =0;

		for (itMask.GoToBegin(), itTens1.GoToBegin(), itTens2.GoToBegin();
				!itMask.IsAtEnd(), !itTens1.IsAtEnd(), !itTens2.IsAtEnd();
				++itMask, ++itTens1, ++itTens2)
		{
			if (itMask.Get() != 0)
			{
						DiffusionTensorType D = itTens1.Get();
						vnl_matrix<RealType> D_mat, D_temp_mat;
						D_mat = ConvertDT2Mat(D);
						vnl_symmetric_eigensystem<RealType> eig(D_mat);
						vnl_vector<RealType> S; S.set_size(3);
						MatrixType R; R.set_size(3,3);

						for (int i=0; i < 3 ; i++)
						{
						S[i] = eig.get_eigenvalue(i);
						R.set_column(i, eig.get_eigenvector(i));

						}
						RealType prod =S[0]*S[1]*S[2];

						if (prod <= 0)
						{
							for (int i=0; i < 3; i++)
							{
								if (S[i] < 0)
								{
									S[i] = -1* S[i];
								}
							}

							vnl_diag_matrix<RealType> S_diag;
							S_diag.set_diagonal(S);
							D_temp_mat = R*S_diag*R.transpose();

							DiffusionTensorType D_new;
							D_new = ConvertMat2DT(D_temp_mat);
							itTens2.Set(D_new);
							std::cout << itMask.GetIndex() <<std::endl;

						}

						else
						{
							itTens2.Set(itTens1.Get());
						}
			}
		}


		return newTensorImage;
}
