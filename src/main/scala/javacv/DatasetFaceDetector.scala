package javacv

import java.awt.image.{BufferedImage, RenderedImage}
import java.awt.{Color, Font}
import java.io.File
import javax.imageio.ImageIO

import org.bytedeco.javacpp.opencv_core.{Mat, Rect}
import org.bytedeco.javacpp.opencv_objdetect.CascadeClassifier
import org.bytedeco.javacpp.{opencv_highgui, opencv_imgproc}

/**
  * Created by gopikrishnan on 5/17/16.
  */
object DatasetFaceDetector extends App {

  val datasetInputPath = "/home/gopikrishnan/Dataset/people/"
  val datasetOutputPath = "/home/gopikrishnan/Dataset/people_output/"

  for(i <- 1 to 50) {
    val imageFilename = s"${datasetInputPath}person_${"%03d".format(i)}.bmp"
    /*args(0)*/
    val mat = opencv_highgui.imread(imageFilename)

    // convert image to greyscale
    val greyMat = new Mat()
    opencv_imgproc.cvtColor(mat, greyMat, opencv_imgproc.CV_BGR2GRAY, 1)
    //ImageIO.write(greyMat.getBufferedImage, "jpg", new File("output_grey.jpg"))

    // equalize histogram
    val equalizedMat = new Mat()
    opencv_imgproc.equalizeHist(greyMat, equalizedMat)
    //ImageIO.write(equalizedMat.getBufferedImage, "jpg", new File("output_equalized.jpg"))

    val faceXml = FaceWebcamDetectorApp.getClass.getClassLoader.getResource("haarcascade_frontalface_alt.xml").getPath

    val bodyXml = FaceWebcamDetectorApp.getClass.getClassLoader.getResource("haarcascade_fullbody.xml").getPath

    val faceCascade = new CascadeClassifier(faceXml)
    val faceRects = new Rect()

    val bodyCascade = new CascadeClassifier(bodyXml)
    val bodyRects = new Rect()

    faceCascade.detectMultiScale(equalizedMat, faceRects)

    bodyCascade.detectMultiScale(equalizedMat, bodyRects)

    val image = mat.getBufferedImage
    val graphics = image.getGraphics
    graphics.setColor(Color.RED)
    for (i <- 0 until faceRects.limit()) {
      val faceRect = faceRects.position(i)
      graphics.drawRect(faceRect.x, faceRect.y, faceRect.width, faceRect.height)
      graphics.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 18))
      graphics.drawString(s"Face $i", faceRect.x, faceRect.y - 20)
    }

    for (i <- 0 until bodyRects.limit()) {
      val bodyRect = bodyRects.position(i)
      graphics.drawRect(bodyRect.x, bodyRect.y, bodyRect.width, bodyRect.height)
      graphics.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 18))
      graphics.drawString(s"Body $i", bodyRect.x, bodyRect.y - 20)
    }

//    val image_with_face = detectFace(image)

    ImageIO.write(image, "bmp", new File(s"${datasetOutputPath}person_output_${"%03d".format(i)}.bmp"))
  }

//  def detectFace(image: BufferedImage):RenderedImage ={
//    val faceXml = FaceWebcamDetectorApp.getClass.getClassLoader.getResource("haarcascade_frontalface_alt.xml").getPath
//
//    val graphics = image.getGraphics
//    graphics.setColor(Color.RED)
//
//    val faceCascade = new CascadeClassifier(faceXml)
//    val faceRects = new Rect()
//
//    faceCascade.detectMultiScale(equalizedMat, faceRects)
//
//    for (i <- 0 until faceRects.limit()) {
//      val faceRect = faceRects.position(i)
//      graphics.drawRect(faceRect.x, faceRect.y, faceRect.width, faceRect.height)
//      graphics.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 18))
//      graphics.drawString(s"Face $i", faceRect.x, faceRect.y - 20)
//    }
//    image
//  }
}
