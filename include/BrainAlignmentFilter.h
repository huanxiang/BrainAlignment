/*==========================================================================================
 *
 *	Program:	TOMI
 *	Module:		BrainAlignment
 *	Language: C++
 *	Data:			$Date: 2014-03-17 10:47:53$
 *	Version:	$Revision: 0.1$
 *
 *	Copyright Philips Research China. All rights reserved.
 *
 *  This program aligns the brain image into the mni atlas space using affine registration.
 *
 *=========================================================================================*/

#ifndef _BrainAlignmentFilter_h_
#define _BrainAlignmentFilter_h_

#include "itkImage.h"
#include "itkAffineTransform.h"
#include "itkVersorRigid3DTransform.h"
#include "itkScaleSkewVersor3DTransform.h"

template<class TImage>
class BrainAlignment
{
public:
	/** Constructor */
	BrainAlignment();

	/** Destructor */
	~BrainAlignment();

public:
	/** Standard class typedefs */
	typedef TImage	ImageType;
	typedef typename ImageType::Pointer		ImagePointerType;
	typedef itk::VersorRigid3DTransform<double>	RigidTransformType;
	typedef itk::AffineTransform<double>	AffineTransformType;
	typedef itk::ScaleSkewVersor3DTransform<double>	RSSTransformType;
	typedef float	InternalPixelType;
	typedef itk::Image< InternalPixelType, 3 >	InternalImageType;
	typedef typename InternalImageType::Pointer		InternalImagePointerType;

	/** Set the atlas path */
	void SetAtlasPath(std::string atlasPath){m_AtlasPath = atlasPath;};

	/** Set the input image */
	void SetInput(ImagePointerType inputImage){m_InputImage = inputImage;};

	/** Set the number of samples for mutual information */
	void SetNumberOfSamples(unsigned int numberOfSamples){m_NumberOfSamples = numberOfSamples;};

	/** Return the pointer of the rigid output image */
	ImagePointerType	GetRigidOutput(){return m_RigidOutputImage;};

	/** Return the pointer of the RSS output image */
	ImagePointerType	GetRSSOutput(){return m_RSSOutputImage;};

	/** Return the pointer of the affine output atlas image */
	ImagePointerType	GetAffineOutput(){return m_AffineOutputImage;};

	/** Return the pointer of the affine output atlas image */
	ImagePointerType	GetAffineMask(){return m_AffineMask;};

	/** Return the pointer of the affine output atlas image */
	InternalImagePointerType	GetAffineWM(){return m_AffineWM;};

	/** Return the pointer of the affine output atlas image */
	InternalImagePointerType	GetAffineGM(){return m_AffineGM;};

	/** Return the pointer of the affine output atlas image */
	InternalImagePointerType	GetAffineCSF(){return m_AffineCSF;};

	/** Return the pointer of the affine label image */
	ImagePointerType	GetAffineLabel(){return m_AffineLabel;};

	/** Return the output transform */
	RigidTransformType::Pointer	GetRigidTransform(){return m_RigidTransform;};

	/** Return the output transform */
	AffineTransformType::Pointer	GetAffineTransform(){return m_AffineTransform;};

	/** Align the atlas image to the input image use first rigid registration 
	 then affine registration*/
	void	Update();

	/** Set Downsample factor */
	void SetDownsampleFactor(float downsampleFactor){m_DownsampleFactor = downsampleFactor;};

	/** Align the atlas image to the input image using rigid+skew+scale transform*/
	void	UpdateRSS();

private:
	/** Atlas path. */
	std::string m_AtlasPath;

	/** Image pointer to the atlas. */
	ImagePointerType m_Atlas;

	/** Image pointer to the input image. */
	ImagePointerType m_InputImage;

	/** Image pointer to the rigidly registered input image. */
	ImagePointerType m_RigidOutputImage;

	/** Image pointer to the affinely registered atlas image. */
	ImagePointerType m_AffineOutputImage;

	/** Image pointer to the RSS registered atlas image. */
	ImagePointerType m_RSSOutputImage;

	/** Image pointer to the affinely registered atlas mask. */
	ImagePointerType m_AffineMask;

	/** Image pointer to the affinely registered atlas white matter. */
	InternalImagePointerType m_AffineWM;

	/** Image pointer to the affinely registered atlas grey matter. */
	InternalImagePointerType m_AffineGM;

	/** Image pointer to the affinely registered atlas CSF. */
	InternalImagePointerType m_AffineCSF;

	/** Image pointer to the affinely registered label image. */
	ImagePointerType m_AffineLabel;

	/** Pointer to the rigid transform. */
	RigidTransformType::Pointer m_RigidTransform;

	/** Pointer to the affine transform. */
	AffineTransformType::Pointer m_AffineTransform;

	/** Pointer to the RSS transform. */
	RSSTransformType::Pointer m_RSSTransform;

	/* Number of samples for mutual information computation */
	unsigned int m_NumberOfSamples;

	/** Downsample the atlas image */
	ImagePointerType DownsampleImage(ImagePointerType image);

	/** Downsample factor */
	float m_DownsampleFactor;


};

#include "BrainAlignmentFilter.cxx"

#endif