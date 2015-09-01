#define TOTAL_PTS 7000000
#include "../inc/MapFilterLR2HR.h"
#include "sobol.cpp"
#include "itkImageFileWriter.h"

unsigned long int MapFilterLR2HR1::ComputeMatrixIndex1(ImageType::Pointer image, ImageType::IndexType index)
{
	ImageType::SizeType size;
	size = image->GetLargestPossibleRegion().GetSize();
	unsigned long int N = index[2]*size[0]*size[1] + index[1]*size[0] +index[0];
	return N;

}

void MapFilterLR2HR1::ReadAffineTransform(AffineTransformType::Pointer transform)
{
	m_AffineTransform = transform;		
}

void MapFilterLR2HR1::ReadHRImage(ImageType::Pointer image)
{
	m_imageHR = image; 
}


void MapFilterLR2HR1::ReadLRImage(ImageType::Pointer image)
{
	m_imageLR = image;
}

void MapFilterLR2HR1::ReadMaskImage(ImageType::Pointer image)
{
	m_MaskImage = image;
}

void MapFilterLR2HR1::ReadDeformationField(DisplacementFieldImageType::Pointer image)
{
	m_DispField = image;
}


void MapFilterLR2HR1::ComputeMapWithDefField()
{
	ImageType::SizeType sizeLR, sizeHR;
	typedef itk::InverseDisplacementFieldImageFilter<DisplacementFieldImageType, DisplacementFieldImageType> InverseDispFieldImageFilterType;
	InverseDispFieldImageFilterType::Pointer invDispFieldFilter = InverseDispFieldImageFilterType::New();
	
	std::cout << "Inverting the DisplacementField " << std::endl;	

//	invDispFieldFilter->SetInput(m_DispField);
	//invDispFieldFilter->Update();
//	DisplacementFieldImageType::Pointer invDispField = invDispFieldFilter->GetOutput();

	DisplacementFieldImageType::IndexType testIndex;
	testIndex[0]=87; testIndex[1]=103; testIndex[2]=128;
	
	DisplacementFieldTransformType::Pointer dispFieldTransform = DisplacementFieldTransformType::New();
	dispFieldTransform->SetDisplacementField(m_DispField);
	dispFieldTransform->GetInverseDisplacementField();
	
	DisplacementFieldImageType::Pointer invDisp = DisplacementFieldImageType::New();
	invDisp = dispFieldTransform->GetInverseDisplacementField();


	sizeLR= m_imageLR->GetLargestPossibleRegion().GetSize();
	sizeHR= m_imageHR->GetLargestPossibleRegion().GetSize();

	unsigned long int row_LR = sizeLR[0]*sizeLR[1]*sizeLR[2];
	unsigned long int col_HR = sizeHR[0]*sizeHR[1]*sizeHR[2];


	vnl_sparse_matrix<float> SpVnl_Row(row_LR, col_HR);
	vnl_sparse_matrix<float> SpVnl_Col(col_HR, row_LR);

	m_Origin_LR = m_imageLR->GetOrigin();
	m_Origin_HR = m_imageHR->GetOrigin();

	ImageType::PointType m_Final_HR, m_Final_LR;
	ImageType::IndexType IndexFinal;

	for (int i =0; i < 3 ; i++)
	{
		IndexFinal[i] = sizeLR[i]-1;
	}

	m_imageLR->TransformIndexToPhysicalPoint(IndexFinal, m_Final_LR);

	for (int num=0; num < TOTAL_PTS; num++)
	{	
		 PointType P_LR, P_HR;
		
		for (int j=0; j < 3 ; j++)
		{
		 double r =  ((double) rand() /(RAND_MAX)) ;
		double temp_add = r*(m_Final_LR[j] - m_Origin_LR[j]);
		     P_LR[j] = m_Origin_LR[j] + temp_add;
			 }

	 	 ImageType::IndexType indexLR, indexHR;

	 	 m_imageLR->TransformPhysicalPointToIndex(P_LR, indexLR );
	   	 std::cout << P_LR << std::endl;

//	   	 P_HR = m_AffineTransform->TransformPoint(P_LR);
	   	 P_HR = dispFieldTransform->TransformPoint(P_LR);
	   	 std::cout << P_HR << std::endl;


		m_imageHR->TransformPhysicalPointToIndex(P_HR, indexHR);

	   	 unsigned long int rN, cN;

	     rN= ComputeMatrixIndex1(m_imageLR, indexLR);
	   	 cN= ComputeMatrixIndex1(m_imageHR, indexHR);
	   	 SpVnl_Row(rN,cN) = SpVnl_Row(rN, cN) +1;


	}

	SpVnl_Col = SpVnl_Row.transpose();

	m_SpVnl_Row_normalized.set_size(row_LR, col_HR);
	m_SpVnl_Col_normalized.set_size(col_HR, row_LR);

    typedef vnl_sparse_matrix_pair<float> pair_t;

	for (int i=0; i < row_LR; i++)
	    {
	        vcl_vector<pair_t> rowM=SpVnl_Row.get_row(i);
	        vcl_vector< int> rowIndices;
	        vcl_vector<float> rowValues;

	        float total_row = SpVnl_Row.sum_row(i);
	        if (total_row == 0)
	        {
	        	total_row =1;
	        }
	        else
	        {
	        for ( vcl_vector<pair_t>::const_iterator it = rowM.begin() ; it !=   rowM.end() ; ++it )
	        {
	            rowIndices.push_back( it->first );
	            rowValues.push_back( it->second/total_row );
	        }
	        }

	        m_SpVnl_Row_normalized.set_row(i, rowIndices, rowValues);
	    }

	    for (int i=0; i < col_HR; i++)
	    {
	        vcl_vector<pair_t> colM=SpVnl_Col.get_row(i);

	        vcl_vector<int> rowIndices;
	        vcl_vector<float> rowValues;

	        float total_row = SpVnl_Col.sum_row(i);


	        if (total_row == 0)
	            total_row=1;
	        for (vcl_vector<pair_t>::const_iterator it = colM.begin(); it != colM.end(); ++it)
	        {
	            rowIndices.push_back( it->first);
	            rowValues.push_back( it->second/total_row);
	        }

	        m_SpVnl_Col_normalized.set_row(i, rowIndices, rowValues);
	    }


}		 


void MapFilterLR2HR1::ComputeMap()
{

	std::cout << "into the filter" << std::endl;
	ImageType::SizeType sizeLR, sizeHR;

	m_AffineTransform->GetInverse(m_AffineTransform) ;
	sizeLR= m_imageLR->GetLargestPossibleRegion().GetSize();
	sizeHR= m_imageHR->GetLargestPossibleRegion().GetSize();

	unsigned long int row_LR = sizeLR[0]*sizeLR[1]*sizeLR[2];
	unsigned long int col_HR = sizeHR[0]*sizeHR[1]*sizeHR[2];

	vnl_sparse_matrix<float> SpVnl_Row(row_LR, col_HR);
	vnl_sparse_matrix<float> SpVnl_Col(col_HR, row_LR);

	m_Origin_LR = m_imageLR->GetOrigin();
	m_Origin_HR = m_imageHR->GetOrigin();

	ImageType::PointType m_Final_HR, m_Final_LR;
	ImageType::IndexType IndexFinal;

	for (int i=0; i < 3; i++)
	{
		IndexFinal[i] = sizeLR[i] - 1;
	}

	m_imageLR->TransformIndexToPhysicalPoint(IndexFinal, m_Final_LR);


	for (int num =0; num < TOTAL_PTS ; num++)
	{
		 PointType P_LR, P_HR;


		 for (int j=0; j< 3; j++)
			 {
			 double r = ((double) rand() / (RAND_MAX)) ;
		     double temp_add = r*(m_Final_LR[j] - m_Origin_LR[j]);
		     P_LR[j] = m_Origin_LR[j] + temp_add;
			 }

	 	 ImageType::IndexType indexLR, indexHR;

	 	 m_imageLR->TransformPhysicalPointToIndex(P_LR, indexLR );

	   	 P_HR = m_AffineTransform->TransformPoint(P_LR);
	   	 m_imageHR->TransformPhysicalPointToIndex(P_HR, indexHR);

	   	 unsigned long int rN, cN;

	     rN= ComputeMatrixIndex1(m_imageLR, indexLR);
	   	 cN= ComputeMatrixIndex1(m_imageHR, indexHR);
	   	 SpVnl_Row(rN,cN) = SpVnl_Row(rN, cN) +1;


	}

	SpVnl_Col = SpVnl_Row.transpose();

	m_SpVnl_Row_normalized.set_size(row_LR, col_HR);
	m_SpVnl_Col_normalized.set_size(col_HR, row_LR);

    typedef vnl_sparse_matrix_pair<float> pair_t;

	for (int i=0; i < row_LR; i++)
	    {
	        vcl_vector<pair_t> rowM=SpVnl_Row.get_row(i);
	        vcl_vector< int> rowIndices;
	        vcl_vector<float> rowValues;

	        float total_row = SpVnl_Row.sum_row(i);
	        if (total_row == 0)
	        {
	        	total_row =1;
	        }
	        else
	        {
	        for ( vcl_vector<pair_t>::const_iterator it = rowM.begin() ; it !=   rowM.end() ; ++it )
	        {
	            rowIndices.push_back( it->first );
	            rowValues.push_back( it->second/total_row );
	        }
	        }

	        m_SpVnl_Row_normalized.set_row(i, rowIndices, rowValues);
	    }

	    for (int i=0; i < col_HR; i++)
	    {
	        vcl_vector<pair_t> colM=SpVnl_Col.get_row(i);

	        vcl_vector<int> rowIndices;
	        vcl_vector<float> rowValues;

	        float total_row = SpVnl_Col.sum_row(i);


	        if (total_row == 0)
	            total_row=1;
	        for (vcl_vector<pair_t>::const_iterator it = colM.begin(); it != colM.end(); ++it)
	        {
	            rowIndices.push_back( it->first);
	            rowValues.push_back( it->second/total_row);
	        }

	        m_SpVnl_Col_normalized.set_row(i, rowIndices, rowValues);
	    }
}


vnl_sparse_matrix<float> MapFilterLR2HR1::GetLR2HRMatrix()
{
	return m_SpVnl_Row_normalized;

}

vnl_sparse_matrix<float> MapFilterLR2HR1::GetHR2LRMatrix()
{
	return m_SpVnl_Col_normalized;
}

