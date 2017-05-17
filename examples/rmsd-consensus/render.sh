spec="consensus"
palette="/tmp/palette.png"
filters="fps=15,scale=320:-1:flags=lanczos"

$HOME/opt/ffmpeg/bin/ffmpeg -v warning -i final.$spec.%05d.tga -vf "$filters,palettegen" -y $palette
$HOME/opt/ffmpeg/bin/ffmpeg -v warning -i final.$spec.%05d.tga -i $palette -lavfi "$filters [x]; [x][1:v] paletteuse" -y $spec.gif
