FROM tensorflow/tensorflow:latest
RUN apt -yy update && apt install -yy imagemagick git curl zbar-tools
RUN pip3 install ImageIO
# modify policy.xml since the picture is veeeery big
ADD ctf/policy.xml /etc/ImageMagick-6/policy.xml
