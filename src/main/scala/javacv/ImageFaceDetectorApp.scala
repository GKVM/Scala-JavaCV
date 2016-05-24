package javacv

import java.awt.{Color, Font}
import java.io.File
import javax.imageio.ImageIO

import org.bytedeco.javacpp.opencv_objdetect.CascadeClassifier
import org.bytedeco.javacpp.{opencv_highgui, opencv_imgproc}
import org.bytedeco.javacpp.opencv_core._
import org.bytedeco.javacpp.opencv_imgproc._

object ImageFaceDetectorApp extends App {
  //  if (args.length != 1) {
  //    sys.error("Argument: image filename")
  //    sys.exit()
  //  }
  val imageFilename = "crowd1.jpg"/*args(0)*/
  //val imageFilename = "person_034.bmp"
  val mat = opencv_highgui.imread(imageFilename)

  // convert image to greyscale
  val greyMat = new Mat()
  opencv_imgproc.cvtColor(mat, greyMat, opencv_imgproc.CV_BGR2GRAY, 1)
  //ImageIO.write(greyMat.getBufferedImage, "jpg", new File("output_grey.jpg"))

  // equalize histogram
  val equalizedMat = new Mat()
  opencv_imgproc.equalizeHist(greyMat, equalizedMat)
  ImageIO.write(equalizedMat.getBufferedImage, "jpg", new File("output_equalized.jpg"))

    val faceXml = FaceWebcamDetectorApp.getClass.getClassLoader.getResource("haarcascade_frontalface_alt.xml").getPath
    val faceCascade = new CascadeClassifier(faceXml)

  val bodyXml = FaceWebcamDetectorApp.getClass.getClassLoader.getResource("haarcascade_upperbody.xml").getPath
  val bodyCascade = new CascadeClassifier(bodyXml)

    val faceRects = new Rect()
    faceCascade.detectMultiScale(equalizedMat, faceRects)

  val bodyRects = new Rect()
  bodyCascade.detectMultiScale(equalizedMat, bodyRects)

  val image = mat.getBufferedImage
  val graphics = image.getGraphics
  graphics.setColor(Color.RED)
    for(i <- 0 until faceRects.limit()) {
      val faceRect = faceRects.position(i)
      graphics.drawRect(faceRect.x, faceRect.y, faceRect.width, faceRect.height)
      graphics.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 18))
      graphics.drawString(s"Face $i", faceRect.x, faceRect.y - 20)
    }
  ImageIO.write(image, "jpg", new File(s"output_face_$imageFilename.jpg"))
  for (i <- 0 until bodyRects.limit()) {
    val bodyRect = bodyRects.position(i)
    graphics.drawRect(bodyRect.x, bodyRect.y, bodyRect.width, bodyRect.height)
    graphics.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 18))
    graphics.drawString(s"Body $i", bodyRect.x, bodyRect.y - 20)
  }
  ImageIO.write(image, "jpg", new File(s"output_face&upperBody$imageFilename.jpg"))
}
