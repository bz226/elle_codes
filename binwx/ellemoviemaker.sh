#!/usr/bin/perl
#
# Script makes gif-movies from elle files
# written by J.K. Becker in 2002
# Uses Perl (what a surprise!), Gtk library, Gtk-Perl, ImageMagick and of course a working
# Elle version.
# Send comments to becker@jkbecker.de (although the script should be simple
# enough to understand and use it as it is) and look for updates at
# http://www.jkbecker.de/geology/elle.


use Gtk;         # load the Gtk-Perl module
# use strict;

# set_locale Gtk;
init Gtk;

$false = 0;
$true = 1;


$filename="";
$ffname=$false;
$loop=1000;  	#for the -loop argument of convert (ImageMagick)
$crop=$false;
$gif=$false;
$jpg=$false;
$ppm=$false;
$rm=$true; 	#for the -crop argument of convert (ImageMagick)
$frame=10;   	#steps in saving a picture in elle.
		#10 means that every 10th timestep a .elle file was saved...
$count=1000; 	#how many timesteps in total did elle make?
		#default here is 1000. together with frame this
		#means that moviemake will look for files with
		#filename010.elle, filenam020.elle ... filname$count.elle

# Create the window
$window = new Gtk::Window( "toplevel" );
$window->signal_connect( "delete_event", sub { Gtk->exit( 0 ); } );
$window->title( "Movie-Maker for Elle" );
$window->border_width( 10 );

#Create frame elle
$frameelle = new Gtk::Frame();
$frameelle->set_label( "Elle-Input" );
$frameelle->set_label_align( 0.0, 0.0 );
$frameelle->set_shadow_type( 'etched_out' );
$frameelle->show();

#create filename widget
$filebutton = new Gtk::Button( "File");
$filebutton->signal_connect ("clicked", \&Enter_Filename);
$filebutton->show();
$filelabel = new Gtk::Label( "None" );
$filelabel->show();

#create frame widget (has nothing to do with frameelle or frameimagick!!!!)
$frameentry= new Gtk::Entry(5);
$frameentry->signal_connect( "activate", \&Enter_Frame, $frameentry );
$frameentry->set_text($frame);
$framelabel = new Gtk::Label( "Frames:" );
$frameentry->show();
$framelabel->show();

#create count widget
$countentry= new Gtk::Entry(5);
$countentry->signal_connect( "activate", \&Enter_Count, $countentry );
$countentry->set_text($count);
$countlabel = new Gtk::Label( "Count:" );
$countentry->show();
$countlabel->show();



#put everything from the frame elle into a vbox and put that into the frame elle
$vbox=new Gtk::VBox($false,0);
$vbox->pack_start_defaults($filebutton);
$vbox->pack_start_defaults($filelabel);
$vbox->pack_start_defaults($framelabel);
$vbox->pack_start_defaults($frameentry);
$vbox->pack_start_defaults($countlabel);
$vbox->pack_start_defaults($countentry);
$vbox->show();
$frameelle->add($vbox);
#-------------------------------------------------------------------------
#Create frame imagick
$frameimagick = new Gtk::Frame();
$frameimagick->set_label( "ImageMagick-Input" );
$frameimagick->set_label_align( 0.0, 0.0 );
$frameimagick->set_shadow_type( 'etched_out' );
$frameimagick->show();

#create loop widget
$loopentry= new Gtk::Entry();
$loopentry->signal_connect( "activate", \&Enter_Loop, $loopentry );
$loopentry->set_text($loop);
$looplabel = new Gtk::Label( "Loop:" );
$loopentry->show();
$looplabel->show();

#Create crop-checkbox
$check = new Gtk::CheckButton( "Crop images?" );
$check->signal_connect( "toggled", \&Entry_Crop, $entry );
$check->set_active( $true );
$check->show();

#Create animated gif
$gifb = new Gtk::CheckButton( "Create animated gif?" );
$gifb->signal_connect( "toggled", \&Entry_gif, $entry );
$gifb->set_active( $true );
$gifb->show();

#Create JPG
$jpgb = new Gtk::CheckButton( "Create JPEG'S?" );
$jpgb->signal_connect( "toggled", \&Entry_jpg, $entry );
$jpgb->set_active( $false );
$jpgb->show();

#Create PPM
$ppmb = new Gtk::CheckButton( "Create PPM'S?" );
$ppmb->signal_connect( "toggled", \&Entry_ppm, $entry );
$ppmb->set_active( $false );
$ppmb->show();

#Remove postscripts?
$rmb = new Gtk::CheckButton( "Remove postscript-files?" );
$rmb->signal_connect( "toggled", \&Entry_rm, $entry );
$rmb->set_active( $true );
$rmb->show();

#put everything from the frame imagick into a vbox and put that into the frame imagick
$vbox1=new Gtk::VBox($false,0);
$vbox1->pack_start_defaults($looplabel);
$vbox1->pack_start_defaults($loopentry);
$vbox1->pack_start_defaults($check);
$vbox1->pack_start_defaults($gifb);
$vbox1->pack_start_defaults($jpgb);
$vbox1->pack_start_defaults($ppmb);
$vbox1->pack_start_defaults($rmb);
$vbox1->show();
$frameimagick->add($vbox1);


#Create "Go" and "Quit" buttons
$gobutton = new Gtk::Button( "Go");
$gobutton->signal_connect ("clicked", \&GoButtonClicked);
$gobutton->show();
$helpbutton = new Gtk::Button( "Help");
$helpbutton->signal_connect ("clicked", \&HelpButtonClicked);
$helpbutton->show();
$quitbutton = new Gtk::Button( "Quit" );
$quitbutton->signal_connect( "clicked", sub { Gtk->exit( 0 ); } );
$quitbutton->show();
$gqhbox=new Gtk::HBox($false,0);
$gqhbox->pack_start_defaults($gobutton);
$gqhbox->pack_start_defaults($quitbutton);
$gqhbox->pack_start_defaults($helpbutton);
$gqhbox->show();

#create text-output window
$textout = new Gtk::Text(undef,undef);
$textout->show();
$vscrollbar = new Gtk::VScrollbar( $textout->vadj );
$vscrollbar->show();


#Show and add everything and put it where it belongs... I lost track here...
$vbox2=new Gtk::VBox($false,0);
$frameelle->show();
$frameimagick->show();
$vbox2->pack_start_defaults($frameelle);
$vbox2->pack_start_defaults($frameimagick);
$vbox2->pack_start_defaults($gqhbox);
$vbox2->show();
$hbox2=new Gtk::HBox($false,0);
$hbox2->pack_start_defaults($textout);
$hbox2->pack_start_defaults($vscrollbar);
$hbox2->show();
$mainhbox=new Gtk::HBox($false,0);
$mainhbox->pack_start_defaults($vbox2);
$mainhbox->pack_start_defaults($hbox2);
$mainhbox->show();
$window->add($mainhbox);
$window->show();

main Gtk;
exit( 0 );


### Subroutines

#New filename was entered
sub Enter_Filename
{
	# Create a new file selection widget
	$file_dialog = new Gtk::FileSelection( "File Selection" );
	$file_dialog->signal_connect( "destroy", sub { $file_dialog->hide(); } );

	# Connect the ok_button to file_ok_sel function
	$file_dialog->ok_button->signal_connect( "clicked",\&file_ok_sel,$file_dialog );

	# Connect the cancel_button to hide the window
	$file_dialog->cancel_button->signal_connect( "clicked",sub { $file_dialog->hide(); } );
	$file_dialog->show();
	# Get the selected filename and remove xxx.elle from it, we will put that back
	# later.....
	sub file_ok_sel
	{
   		my ( $widget, $file_selection ) = @_;
   		$filename = $file_selection->get_filename();
		$len=length $filename; 			#get length of string
		$len=$len-8;
		$filename=substr( $filename, 0, $len);	#remove "xxx.elle" from the string
		$filelabel->set_text( $filename."xxx.elle");#show filename but with "xxx.elle"
		$ffname=$true;
		$file_dialog->hide();
	}
}

#New loop number was entered
sub Enter_Loop
{
   my ( $widget, $entry ) = @_;
   $loop=$entry->get_text();
}

#Help button clicked
sub HelpButtonClicked
{
	$textout->insert( undef, $textout->style->black, undef,"Button File: A window pops up where you have to choose the filename of the simulation ending with 010.elle\n\nFrames: Corresponds to the setting SaveInterval in the Elle simulations.\n\nCount:Corresponds to Stages in the Elle simulation\n\nLoop: For the convert-command of ImageMagick. How often should the gif-movie loop.\n\nCrop images?: For the convert-command of ImageMagick. Do you want to clip out unused areas of the image in the gif-movie.\n\n");
}

#crop or not, the answer my friend is blowing in the wind
sub Entry_Crop
{
  if($crop==$true)
  {$crop=$false;}
  else
  {$crop=$true;}
}

sub Entry_gif
{
  if($gif==$true)
  {$gif=$false;}
  else
  {$gif=$true;}
}
sub Entry_jpg
{
  if($jpg==$true)
  {$jpg=$false;}
  else
  {$jpg=$true;}
}

sub Entry_ppm
{
  if($ppm==$true)
  {$ppm=$false;}
  else
  {$ppm=$true;}
}
sub Entry_rm
{
  if($rm==$true)
  {$rm=$false;}
  else
  {$rm=$true;}
}
#What is the distance of the frames?
sub Enter_Frame
{
     my ( $widget, $entry ) = @_;
     $frame=$entry->get_text();
}

#How many files are there
sub Enter_Count
{
     my ( $widget, $entry ) = @_;
     $count=$entry->get_text();
}

#The Go-button was clicked so here we go
sub GoButtonClicked
{
	#$filename holds the filename without xxx.elle
	#$frame holds an integer of the steps used for saving in elle
	#$count holds an integer of how many timesteps were done in total with elle
	#$loop holds an integer for convert (ImageMagick)
	#$crop is true (1) if user wants to crop image
	$frameentry->activate();
	$loopentry->activate();
	$countentry->activate();
	$name=$filename;
	if ($ffname==$true)
	{
		#and now we will perform a small trick, we will rename all the elle-files. Otherwise, the
		#convert command from imagemagick produces funny stuff.... after this renaming, all
		#files will be in the format xxx.elle instead of NAMExxx.elle
		#they will be deleted later
		#
		#first we have to find the name without the path
		$len=length $filename; 		#get length of string
		#printf("filename=$filename\n");
		#printf("Länge des strings: $len\n");
		for ($i=0;$i<=$len;$i++)
		{
		#	printf("i=$i\n");
			$test=substr ($filename,$i,1);
		#	printf("test=$test\n");
			if ($test eq "/")
			{
				$length=$i+1;
		#		printf("found one bei $length \n");
			}
		}
		$path=substr( $filename, 0, $length); #this is only the path
		#printf("Pfad=$path\n");
		$name=substr ($filename,$length, $len);	#this is only the name WITHOUT xxx.elle at the end
		#printf("NAME=$name\n");
		for ($i=$frame;$i<=$count;$i=$i+$frame) #use elle-files with showelleps
		{
			if($i<100)
			{
				$filename= $path .$name."0". $i . ".elle"; #generate filename with number
				system("showelleps -i $filename"); 	#execute showelleps
				$string="Using file: ".$filename."\n";
				$textout->insert( undef, $textout->style->black, undef,$string);
			}
			else
			{
				$filename= $path .$name. $i . ".elle"; #generate filename with number
				system("showelleps -i $filename"); 	#execute showelleps
				$string="Using file: ".$filename."\n";
				$textout->insert( undef, $textout->style->black, undef,$string);
			}
		}
		$namegif=$path . $name . ".gif";
		$string="Making movie: ". $namegif. "\n";
		$textout->insert( undef, $textout->style->black, undef,$string);
		$textout->insert( undef, $textout->style->black, undef,"This might take a while....\n");
		if($jpg==$true)
		{
			for ($i=$frame;$i<=$count;$i=$i+$frame)
			{
				if($i<100)
					{
						$namejpg=$path.$name."0".$i.".jpg";
						$filename=$path .$name."0". $i . ".ps";
						system("convert -crop 0x0 $filename $namejpg"); #convert all ps files into one gif-movie
						$cmap = Gtk::Gdk::Colormap->get_system();
						$color->{ 'green' } = 0xFFFF;
						$textout->insert( undef, $color, undef,$namejpg."finished\n");
					}
					else
					{
						$namejpg=$path.$name.$i.".jpg";
						$filename=$path .$name. $i . ".ps";
						system("convert -crop 0x0 $filename $namejpg"); #convert all ps files into one gif-movie
						$cmap = Gtk::Gdk::Colormap->get_system();
						$color->{ 'green' } = 0xFFFF;
						$textout->insert( undef, $color, undef,$namejpg."finished\n");
					}
			}
		}
		if($ppm==$true)
		{
			$x=0;
			for ($i=$frame;$i<=$count;$i=$i+$frame)
			{
				if($i<100)
					{
						$namejpg=$path.$name.$x.".ppm";
						$filename=$path .$name."0". $i . ".ps";
						system("convert -crop 0x0 $filename $namejpg"); #convert all ps files into one gif-movie
						$cmap = Gtk::Gdk::Colormap->get_system();
						$color->{ 'green' } = 0xFFFF;
						$textout->insert( undef, $color, undef,$namejpg."finished\n");
						$x=$x+1;
					}
					else
					{
						$namejpg=$path.$name.$x.".ppm";
						$filename=$path .$name. $i . ".ps";
						system("convert -crop 0x0 $filename $namejpg"); #convert all ps files into one gif-movie
						$cmap = Gtk::Gdk::Colormap->get_system();
						$color->{ 'green' } = 0xFFFF;
						$textout->insert( undef, $color, undef,$namejpg."finished\n");
						$x=$x+1;
					}
			}
		}
		if($gif==$true)
		{
			if ($crop==$true)
			{
				$filename=$path .$name. $count . ".ps";
				system("convert -loop $loop -crop 0x0 $filename $namegif");
				for ($i=$count;$i>=$frame;$i=$i-$frame)#delete new elle-files that only have numbers
				{
					if($i<100)
					{
						$filename=$path .$name."0". $i . ".ps";
						system("convert -loop $loop -crop 0x0 $filename $namegif $namegif"); #convert all ps files into one gif-movie
						$cmap = Gtk::Gdk::Colormap->get_system();
						$color->{ 'green' } = 0xFFFF;
						$textout->insert( undef, $color, undef,$filename."finished\n");
					}
					else
					{
						$filename=$path .$name. $i . ".ps";
						system("convert -loop $loop -crop 0x0 $filename $namegif $namegif"); #convert all ps files into one gif-movie
						$cmap = Gtk::Gdk::Colormap->get_system();
						$color->{ 'green' } = 0xFFFF;
						$textout->insert( undef, $color, undef,$filename."finished\n");
					}
				}
			}
			else
			{
				$filename=$path .$name. $count . ".ps";
				system("convert -loop $loop $filename $namegif");
				for ($i=$count;$i>=$frame;$i=$i-$frame)#delete new elle-files that only have numbers
				{
					if($i<100)
					{
						$filename=$path .$name."0". $i . ".ps";
						system("convert -loop $loop $filename $namegif $namegif"); #convert all ps files into one gif-movie
						$cmap = Gtk::Gdk::Colormap->get_system();
						$color->{ 'green' } = 0xFFFF;
						$textout->insert( undef, $color, undef,$filename."finished\n");
					}
					else
					{
						$filename=$path .$name. $i . ".ps";
						system("convert -loop $loop $filename $namegif $namegif"); #convert all ps files into one gif-movie
						$cmap = Gtk::Gdk::Colormap->get_system();
						$color->{ 'green' } = 0xFFFF;
						$textout->insert( undef, $color, undef,$filename."finished\n");
					}
				}
			}
		}
		if($rm==$true)
		{
			system("rm *.ps");
			$cmap = Gtk::Gdk::Colormap->get_system();
			$color->{ 'green' } = 0xFFFF;
			$textout->insert( undef, $color, undef,"Removed ps-files\n");
		}
		$textout->insert( undef, $color, undef,"Everything finished\n");
	}
	else
	{
		$cmap = Gtk::Gdk::Colormap->get_system();
		$color->{ 'red' } = 0xFFFF;
		$textout->insert( undef, $color, undef, "I really do need a filename to work with! Please choose one!\n");
	}
}
