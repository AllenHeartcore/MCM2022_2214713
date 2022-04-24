#!/bin/bash

clear
echo "<<< lc3tools Installer >>>"

echo -e "\nInstalling support libraries..."
sudo apt -y install flex wish libreadline6-dev libncurses5-dev &> /dev/null

echo -e "\nWriting source files..."
mkdir lc3tools
cd lc3tools
cat>COPYING<<EOF
                    GNU GENERAL PUBLIC LICENSE
                       Version 2, June 1991

 Copyright (C) 1989, 1991 Free Software Foundation, Inc.
                       59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.

                            Preamble

  The licenses for most software are designed to take away your
freedom to share and change it.  By contrast, the GNU General Public
License is intended to guarantee your freedom to share and change free
software--to make sure the software is free for all its users.  This
General Public License applies to most of the Free Software
Foundation's software and to any other program whose authors commit to
using it.  (Some other Free Software Foundation software is covered by
the GNU Library General Public License instead.)  You can apply it to
your programs, too.

  When we speak of free software, we are referring to freedom, not
price.  Our General Public Licenses are designed to make sure that you
have the freedom to distribute copies of free software (and charge for
this service if you wish), that you receive source code or can get it
if you want it, that you can change the software or use pieces of it
in new free programs; and that you know you can do these things.

  To protect your rights, we need to make restrictions that forbid
anyone to deny you these rights or to ask you to surrender the rights.
These restrictions translate to certain responsibilities for you if you
distribute copies of the software, or if you modify it.

  For example, if you distribute copies of such a program, whether
gratis or for a fee, you must give the recipients all the rights that
you have.  You must make sure that they, too, receive or can get the
source code.  And you must show them these terms so they know their
rights.

  We protect your rights with two steps: (1) copyright the software, and
(2) offer you this license which gives you legal permission to copy,
distribute and/or modify the software.

  Also, for each author's protection and ours, we want to make certain
that everyone understands that there is no warranty for this free
software.  If the software is modified by someone else and passed on, we
want its recipients to know that what they have is not the original, so
that any problems introduced by others will not reflect on the original
authors' reputations.

  Finally, any free program is threatened constantly by software
patents.  We wish to avoid the danger that redistributors of a free
program will individually obtain patent licenses, in effect making the
program proprietary.  To prevent this, we have made it clear that any
patent must be licensed for everyone's free use or not licensed at all.

  The precise terms and conditions for copying, distribution and
modification follow.

                    GNU GENERAL PUBLIC LICENSE
   TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION

  0. This License applies to any program or other work which contains
a notice placed by the copyright holder saying it may be distributed
under the terms of this General Public License.  The "Program", below,
refers to any such program or work, and a "work based on the Program"
means either the Program or any derivative work under copyright law:
that is to say, a work containing the Program or a portion of it,
either verbatim or with modifications and/or translated into another
language.  (Hereinafter, translation is included without limitation in
the term "modification".)  Each licensee is addressed as "you".

Activities other than copying, distribution and modification are not
covered by this License; they are outside its scope.  The act of
running the Program is not restricted, and the output from the Program
is covered only if its contents constitute a work based on the
Program (independent of having been made by running the Program).
Whether that is true depends on what the Program does.

  1. You may copy and distribute verbatim copies of the Program's
source code as you receive it, in any medium, provided that you
conspicuously and appropriately publish on each copy an appropriate
copyright notice and disclaimer of warranty; keep intact all the
notices that refer to this License and to the absence of any warranty;
and give any other recipients of the Program a copy of this License
along with the Program.

You may charge a fee for the physical act of transferring a copy, and
you may at your option offer warranty protection in exchange for a fee.

  2. You may modify your copy or copies of the Program or any portion
of it, thus forming a work based on the Program, and copy and
distribute such modifications or work under the terms of Section 1
above, provided that you also meet all of these conditions:

    a) You must cause the modified files to carry prominent notices
    stating that you changed the files and the date of any change.

    b) You must cause any work that you distribute or publish, that in
    whole or in part contains or is derived from the Program or any
    part thereof, to be licensed as a whole at no charge to all third
    parties under the terms of this License.

    c) If the modified program normally reads commands interactively
    when run, you must cause it, when started running for such
    interactive use in the most ordinary way, to print or display an
    announcement including an appropriate copyright notice and a
    notice that there is no warranty (or else, saying that you provide
    a warranty) and that users may redistribute the program under
    these conditions, and telling the user how to view a copy of this
    License.  (Exception: if the Program itself is interactive but
    does not normally print such an announcement, your work based on
    the Program is not required to print an announcement.)

These requirements apply to the modified work as a whole.  If
identifiable sections of that work are not derived from the Program,
and can be reasonably considered independent and separate works in
themselves, then this License, and its terms, do not apply to those
sections when you distribute them as separate works.  But when you
distribute the same sections as part of a whole which is a work based
on the Program, the distribution of the whole must be on the terms of
this License, whose permissions for other licensees extend to the
entire whole, and thus to each and every part regardless of who wrote it.

Thus, it is not the intent of this section to claim rights or contest
your rights to work written entirely by you; rather, the intent is to
exercise the right to control the distribution of derivative or
collective works based on the Program.

In addition, mere aggregation of another work not based on the Program
with the Program (or with a work based on the Program) on a volume of
a storage or distribution medium does not bring the other work under
the scope of this License.

  3. You may copy and distribute the Program (or a work based on it,
under Section 2) in object code or executable form under the terms of
Sections 1 and 2 above provided that you also do one of the following:

    a) Accompany it with the complete corresponding machine-readable
    source code, which must be distributed under the terms of Sections
    1 and 2 above on a medium customarily used for software interchange; or,

    b) Accompany it with a written offer, valid for at least three
    years, to give any third party, for a charge no more than your
    cost of physically performing source distribution, a complete
    machine-readable copy of the corresponding source code, to be
    distributed under the terms of Sections 1 and 2 above on a medium
    customarily used for software interchange; or,

    c) Accompany it with the information you received as to the offer
    to distribute corresponding source code.  (This alternative is
    allowed only for noncommercial distribution and only if you
    received the program in object code or executable form with such
    an offer, in accord with Subsection b above.)

The source code for a work means the preferred form of the work for
making modifications to it.  For an executable work, complete source
code means all the source code for all modules it contains, plus any
associated interface definition files, plus the scripts used to
control compilation and installation of the executable.  However, as a
special exception, the source code distributed need not include
anything that is normally distributed (in either source or binary
form) with the major components (compiler, kernel, and so on) of the
operating system on which the executable runs, unless that component
itself accompanies the executable.

If distribution of executable or object code is made by offering
access to copy from a designated place, then offering equivalent
access to copy the source code from the same place counts as
distribution of the source code, even though third parties are not
compelled to copy the source along with the object code.

  4. You may not copy, modify, sublicense, or distribute the Program
except as expressly provided under this License.  Any attempt
otherwise to copy, modify, sublicense or distribute the Program is
void, and will automatically terminate your rights under this License.
However, parties who have received copies, or rights, from you under
this License will not have their licenses terminated so long as such
parties remain in full compliance.

  5. You are not required to accept this License, since you have not
signed it.  However, nothing else grants you permission to modify or
distribute the Program or its derivative works.  These actions are
prohibited by law if you do not accept this License.  Therefore, by
modifying or distributing the Program (or any work based on the
Program), you indicate your acceptance of this License to do so, and
all its terms and conditions for copying, distributing or modifying
the Program or works based on it.

  6. Each time you redistribute the Program (or any work based on the
Program), the recipient automatically receives a license from the
original licensor to copy, distribute or modify the Program subject to
these terms and conditions.  You may not impose any further
restrictions on the recipients' exercise of the rights granted herein.
You are not responsible for enforcing compliance by third parties to
this License.

  7. If, as a consequence of a court judgment or allegation of patent
infringement or for any other reason (not limited to patent issues),
conditions are imposed on you (whether by court order, agreement or
otherwise) that contradict the conditions of this License, they do not
excuse you from the conditions of this License.  If you cannot
distribute so as to satisfy simultaneously your obligations under this
License and any other pertinent obligations, then as a consequence you
may not distribute the Program at all.  For example, if a patent
license would not permit royalty-free redistribution of the Program by
all those who receive copies directly or indirectly through you, then
the only way you could satisfy both it and this License would be to
refrain entirely from distribution of the Program.

If any portion of this section is held invalid or unenforceable under
any particular circumstance, the balance of the section is intended to
apply and the section as a whole is intended to apply in other
circumstances.

It is not the purpose of this section to induce you to infringe any
patents or other property right claims or to contest validity of any
such claims; this section has the sole purpose of protecting the
integrity of the free software distribution system, which is
implemented by public license practices.  Many people have made
generous contributions to the wide range of software distributed
through that system in reliance on consistent application of that
system; it is up to the author/donor to decide if he or she is willing
to distribute software through any other system and a licensee cannot
impose that choice.

This section is intended to make thoroughly clear what is believed to
be a consequence of the rest of this License.

  8. If the distribution and/or use of the Program is restricted in
certain countries either by patents or by copyrighted interfaces, the
original copyright holder who places the Program under this License
may add an explicit geographical distribution limitation excluding
those countries, so that distribution is permitted only in or among
countries not thus excluded.  In such case, this License incorporates
the limitation as if written in the body of this License.

  9. The Free Software Foundation may publish revised and/or new versions
of the General Public License from time to time.  Such new versions will
be similar in spirit to the present version, but may differ in detail to
address new problems or concerns.

Each version is given a distinguishing version number.  If the Program
specifies a version number of this License which applies to it and "any
later version", you have the option of following the terms and conditions
either of that version or of any later version published by the Free
Software Foundation.  If the Program does not specify a version number of
this License, you may choose any version ever published by the Free Software
Foundation.

  10. If you wish to incorporate parts of the Program into other free
programs whose distribution conditions are different, write to the author
to ask for permission.  For software which is copyrighted by the Free
Software Foundation, write to the Free Software Foundation; we sometimes
make exceptions for this.  Our decision will be guided by the two goals
of preserving the free status of all derivatives of our free software and
of promoting the sharing and reuse of software generally.

EOF
cat>NO_WARRANTY<<EOF
                            NO WARRANTY

  11. BECAUSE THE PROGRAM IS LICENSED FREE OF CHARGE, THERE IS NO WARRANTY
FOR THE PROGRAM, TO THE EXTENT PERMITTED BY APPLICABLE LAW.  EXCEPT WHEN
OTHERWISE STATED IN WRITING THE COPYRIGHT HOLDERS AND/OR OTHER PARTIES
PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESSED
OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  THE ENTIRE RISK AS
TO THE QUALITY AND PERFORMANCE OF THE PROGRAM IS WITH YOU.  SHOULD THE
PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF ALL NECESSARY SERVICING,
REPAIR OR CORRECTION.

  12. IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING
WILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MAY MODIFY AND/OR
REDISTRIBUTE THE PROGRAM AS PERMITTED ABOVE, BE LIABLE TO YOU FOR DAMAGES,
INCLUDING ANY GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING
OUT OF THE USE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED
TO LOSS OF DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY
YOU OR THIRD PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER
PROGRAMS), EVEN IF SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE
POSSIBILITY OF SUCH DAMAGES.
EOF
cat>README<<EOF
-----------------------------
   DESCRIPTION OF CONTENTS
-----------------------------

The LC-3 tools package contains the lc3as assembler, the lc3sim simulator,
and lc3sim-tk, a Tcl/Tk-based GUI frontend to the simulator.  All tools,
code, etc., were developed by Steven S. Lumetta on his own time with his
own resources for use in teaching classes based on the textbook by 
Yale N. Patt and Sanjay J. Patel entitled, "Introduction to Computing
Systems: From Bits & Gates to C & Beyond," second edition, McGraw-Hill,
New York, New York, 2004, ISBN-0-07-246750-9.

The contents of the LC-3 tools distribution, including sources, management
tools, and data, are Copyright (c) 2003 Steven S. Lumetta.

The LC-3 tools distribution is free software covered by the GNU General 
Public License, and you are welcome to modify it and/or distribute copies 
of it under certain conditions.  The file COPYING (distributed with the
tools) specifies those conditions.  There is absolutely no warranty for 
the LC-3 tools distribution, as described in the file NO_WARRANTY (also
distributed with the tools).



---------------------
   NECESSARY TOOLS
---------------------

Installation requires versions of gcc (the Gnu C compiler),
flex (Gnu's version of lex), and wish (the Tcl/Tk graphical shell).
All of these tools are available for free from many sources.
If you have Gnu's readline installed, the configure script should
find it and use it for the command line version of the simulator.
I don't currently use the history feature, but will add it...someday.

Other necessary but more standard tools include uname, rm, cp, mkdir, 
and chmod.

Currently, the configure script searches only a few directories.
If your binaries are in a reasonable place that I overlooked, send
me a note and I'll add it to the default list.  If you have 
idiosyncratic path names (e.g., the name of your fileserver in your
path), you will have to add the correct paths to the search path at 
the top of the configure script yourself.

N.B.  I have installed the package on Cygwin, Solaris, and Linux
machines.  Linux has been used by 2-3 other schools at the time of
the 0.5 release; Cygwin is stable on my home machine; Solaris GUI
version caused me grief last time I launched it, but I haven't
had time to investigate.

DEBIAN/UBUNTU USERS (and possibly some other distributions of Linux):
After you configure, remove "-lcurses" from the OS_SIM_LIBS
definition in the Makefile.  (Or you can install the curses library,
but the routines that I use are in the standard library in the
Debian distribution.  In other distributions, they're in the
curses library.  One day, I'll extend configure to check for it.)

MAC USERS may want to install into /usr/local/bin using the
--installdir option described below, as that directory is normally
included in users' default PATH variables.


-------------------------------
   INSTALLATION INSTRUCTIONS
-------------------------------

The LC-3 tools package is designed to work as either a personal or 
administrative installation on various flavors of Unix, including
Windows NT-based systems with appropriate support (e.g., Cygwin).

First, decide where the binaries and LC-3 OS code should be installed.
    * If you want it in the directory in which you unpacked the code,
      simply type "configure."
    * If you want it in a different directory, say /usr/bin, type
      configure --installdir /usr/bin
      replacing /usr/bin with the desired directory.

Then type 'make'.

If you want to make the tools available to other people, next type
'make install'.  If not, don't.

Please send any comments or bug reports to me at lumetta@illinois.edu.
Unfortunately, due to the volume of e-mail that I receive on a regular
basis, I cannot guarantee that I will respond to your mail, but
I will almost definitely read it.

EOF
cat>configure<<EOF
#!/bin/sh

# Parse the arguments...

INSTALL_DIR=\`pwd\`
case \$1 in 
	--installdir) 
	    INSTALL_DIR=\$2 ;
	    ;;
	--help)
	    echo "--installdir <directory in which to install>"
	    echo "--help"
	    exit
	    ;;
esac


# Some binaries that we'll need, and the places that we might find them.

binlist="uname flex gcc wish rm cp mkdir chmod sed"
pathlist="/bin /usr/bin /usr/local/bin /sw/bin /usr/x116/bin /usr/X11R6/bin"
libpathlist="/lib /usr/lib /usr/local/lib"
incpathlist="/include /usr/include /usr/local/include"


# Find the binaries (or die trying).

for binary in \$binlist ; do
    for path in \$pathlist ; do
	if [ -r \$path/\$binary ] ; then
	    eval "\$binary=\${path}/\${binary}" ;
	    break ;
	fi ;
    done ;
    eval "if [ -z \\"\\\$\$binary\\" ] ; then echo \\"Cannot locate \$binary binary.\\" ; exit ; fi"
done


# These default values are overridden below for some operating systems.

OS_SIM_LIBS=""
EXE=""
DYN="so"
CODE_FONT="{{Lucida Console} 11 bold}"
BUTTON_FONT="{{Lucida Console} 10 normal}"
CONSOLE_FONT="{{Lucida Console} 10 bold}"


# Tailor the variables based on OS.

case \`\$uname -s\` in
	CYGWIN*) 
		EXE=".exe"
		DYN="dll"
		echo "Configuring for Cygwin..."
		;;
	Linux*) echo "Configuring for Linux..."
		OS_SIM_LIBS="-lcurses"
		;;
	SunOS*)  echo "Configuring for Solaris..."
		OS_SIM_LIBS="-lcurses -lsocket -lnsl"
		;;
	Darwin*)
		FONT="Fixed"
		if [ "\$wish" = "/sw/bin/wish" ] ; then
		    echo "Configuring for MacOS-X (Darwin)/Fink Tcl/Tk."
		    # Fink installation--override default fonts using
		    # fonts suggested by Tevis Money.
		    CODE_FONT="{LucidaTypewriter 11 normal}"
		    BUTTON_FONT="{Fixed 10 normal}"
		    CONSOLE_FONT="{Fixed 10 normal}"
		else
		    echo "Configuring for MacOS-X (Darwin)/Aqua Tcl/Tk."
		fi
esac
echo "Installation directory is \$INSTALL_DIR"


# Look around for readline.

USE_READLINE=-DUSE_READLINE=1

for path in \$libpathlist ; do
    if [ -r \$path/libreadline.a ] ; then
    	RLLPATH="-L\$path -lreadline" ;
	break ;
    fi ;
    if [ -r \$path/libreadline.\${DYN}.a ] ; then
    	RLLPATH="-L\$path -lreadline" ;
	break ;
    fi ;
done
if [ -z "\$RLLPATH" ] ; then
    USE_READLINE= ;
fi

for path in \$incpathlist ; do
    if [ -d \$path/readline ] ; then
    	RLIPATH=-I\$path ;
	break ;
    fi ;
done
if [ -z "\$RLIPATH" ] ; then
    USE_READLINE= ;
fi


# Splice it all in to Makefile.def to create the Makefile.

rm -f Makefile
sed -e "s __GCC__ \$gcc g" -e "s __FLEX__ \$flex g" -e "s __EXE__ \$EXE g"     \\
    -e "s*__OS_SIM_LIBS__*\$OS_SIM_LIBS*g" -e "s __RM__ \$rm g"               \\
    -e "s __CP__ \$cp g" -e "s __MKDIR__ \$mkdir g" -e "s __CHMOD__ \$chmod g" \\
    -e "s __USE_READLINE__ \$USE_READLINE g" -e "s*__RLLPATH__*\$RLLPATH*g"   \\
    -e "s __RLIPATH__ \$RLIPATH g" -e "s*__INSTALL_DIR__*\$INSTALL_DIR*g"     \\
    -e "s __WISH__ \$wish g" -e "s __SED__ \$sed g"                           \\
    -e "s!__CODE_FONT__!\$CODE_FONT!g" -e "s!__BUTTON_FONT__!\$BUTTON_FONT!g" \\
    -e "s!__CONSOLE_FONT__!\$CONSOLE_FONT!g" Makefile.def > Makefile

EOF
cat>lc3.def<<EOF
/*									tab:8
 *
 * lc3.def - definition file for the LC-3 ISA
 *
 * "Copyright (c) 2003-2020 by Steven S. Lumetta and LIU Tingkai."
 *
 * Permission to use, copy, modify, and distribute this software and its
 * documentation for any purpose, without fee, and without written 
 * agreement is hereby granted, provided that the above copyright notice
 * and the following two paragraphs appear in all copies of this software,
 * that the files COPYING and NO_WARRANTY are included verbatim with
 * any distribution, and that the contents of the file README are included
 * verbatim as part of a file named README with any distribution.
 * 
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE TO ANY PARTY FOR DIRECT, 
 * INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT 
 * OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE AUTHOR 
 * HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 * THE AUTHOR SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT 
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR 
 * A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED HEREUNDER IS ON AN "AS IS" 
 * BASIS, AND THE AUTHOR NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, 
 * UPDATES, ENHANCEMENTS, OR MODIFICATIONS."
 *
 * Author:	    Steve Lumetta
 * Version:	    2
 * Creation Date:   18 October 2003
 * Filename:	    lc3.def
 * History:
 *	SSL	1	18 October 2003
 *		Copyright notices and Gnu Public License marker added.
 *	TKL/SSL	2	9 October 2020
 *		Integrated Tingkai Liu's extensions into main code.
 */



/* Field access macros for instruction code. */

#define INST	REG (R_IR)

#define I_DR    F_DR (INST)
#define I_SR1   F_SR1 (INST)
#define I_SR2   F_SR2 (INST)
#define I_BaseR I_SR1
#define I_SR    I_DR    /* for stores */
#define I_CC    F_CC (INST)
#define I_vec8  F_vec8 (INST)

#define I_imm5  F_imm5 (INST)
#define I_imm6  F_imm6 (INST)
#define I_imm9  F_imm9 (INST)
#define I_imm11 F_imm11 (INST)


/* Macro to set condition codes (used in instruction code). */

#define SET_CC() {                  \\
    REG (R_PSR) &= ~0x0E00;         \\
    if ((REG (I_DR) & 0x8000) != 0) \\
	REG (R_PSR) |= 0x0800;      \\
    else if (REG (I_DR) == 0)       \\
	REG (R_PSR) |= 0x0400;      \\
    else                            \\
	REG (R_PSR) |= 0x0200;      \\
}


/*
 * Instruction definition macro format
 * 
 * DEF_INST(name,format,mask,match,flags,code)
 *
 * name   -- mnemonic operand name for disassembly
 * format -- instruction format (operands to print in disassembly)
 * mask   -- bit vector of bits that must match for this instruction
 * match  -- values of bits to match in instruction
 * flags  -- flags for instruction types
 * code   -- operations to execute for instruction
 *
 *
 * 
 * Pseudo-op definition macro format (disassembly only)
 *
 * DEF_POP(name,format,mask,match)   fields are same as DEF_INST above
 *
 */

DEF_INST (ADD, FMT_RRR, 0xF038, 0x1000, FLG_NONE, {
    RECORD_REG_DELTA (0, I_DR, REG (I_DR));
    RECORD_REG_DELTA (1, R_PSR, REG (R_PSR));
    
    REG (I_DR) = (REG (I_SR1) + REG (I_SR2)) & 0xFFFF;
    SET_CC ();
}); 

DEF_INST (ADD, FMT_RRI, 0xF020, 0x1020, FLG_NONE, {
    RECORD_REG_DELTA (0, I_DR, REG (I_DR));
    RECORD_REG_DELTA (1, R_PSR, REG (R_PSR));
        
    REG (I_DR) = (REG (I_SR1) + I_imm5) & 0xFFFF;
    SET_CC ();
});

DEF_INST (AND, FMT_RRR, 0xF038, 0x5000, FLG_NONE, {
    RECORD_REG_DELTA (0, I_DR, REG (I_DR));
    RECORD_REG_DELTA (1, R_PSR, REG (R_PSR));
        
    REG (I_DR) = REG (I_SR1) & REG (I_SR2);
    SET_CC ();
});

DEF_INST (AND, FMT_RRI, 0xF020, 0x5020, FLG_NONE, {
    RECORD_REG_DELTA (0, I_DR, REG (I_DR));
    RECORD_REG_DELTA (1, R_PSR, REG (R_PSR));
        
    REG (I_DR) = REG (I_SR1) & I_imm5;
    SET_CC ();
});

DEF_P_OP (NOP, FMT_, 0xFFFF, 0x0000);
DEF_P_OP (.FILL, FMT_A, 0xFF00, 0x0000);
DEF_P_OP (NOP, FMT_, 0xF1FF, 0x0000);
DEF_P_OP (NOP, FMT_, 0xFE00, 0x0000);

DEF_INST (BR, FMT_CL, 0xF000, 0x0000, FLG_NONE, {
    if ((REG (R_PSR) & I_CC) != 0)
	REG (R_PC) = (REG (R_PC) + I_imm9) & 0xFFFF;
});

DEF_P_OP (RET, FMT_, 0xFFFF, 0xC1C0);

DEF_INST (JMP, FMT_R, 0xFE3F, 0xC000, FLG_NONE, {
    if (I_BaseR == R_R7) {
        ADD_FLAGS (FLG_RETURN);
    }
    REG (R_PC) = REG (I_BaseR);
});

DEF_INST (JSR, FMT_L, 0xF800, 0x4800, FLG_SUBROUTINE, {
    RECORD_REG_DELTA (0, R_R7, REG (R_R7));
    
    REG (R_R7) = REG (R_PC);
    REG (R_PC) = (REG (R_PC) + I_imm11) & 0xFFFF;
});

/* JSRR -- note that definition does not match second edition of book,
   but intention is to change in 3rd+ printing or 3rd edition. */
DEF_INST (JSRR, FMT_R, 0xFE3F, 0x4000, FLG_SUBROUTINE, {
    RECORD_REG_DELTA (0, R_R7, REG (R_R7));
    
    int tmp = REG (I_BaseR);
    REG (R_R7) = REG (R_PC);
    REG (R_PC) = tmp;
});

DEF_INST (LD, FMT_RL, 0xF000, 0x2000, FLG_NONE, {
    RECORD_REG_DELTA (0, I_DR, REG (I_DR));
    RECORD_REG_DELTA (1, R_PSR, REG (R_PSR));
        
    REG (I_DR) = read_memory ((REG (R_PC) + I_imm9) & 0xFFFF);
    SET_CC ();
});

DEF_INST (LDI, FMT_RL, 0xF000, 0xA000, FLG_NONE, {
    RECORD_REG_DELTA (0, I_DR, REG (I_DR));
    RECORD_REG_DELTA (1, R_PSR, REG (R_PSR));
        
    REG (I_DR) = read_memory (read_memory ((REG (R_PC) + I_imm9) & 0xFFFF));
    SET_CC ();
});

DEF_INST (LDR, FMT_RRI6, 0xF000, 0x6000, FLG_NONE, {
    RECORD_REG_DELTA (0, I_DR, REG (I_DR));
    RECORD_REG_DELTA (1, R_PSR, REG (R_PSR));
        
    REG (I_DR) = read_memory ((REG (I_BaseR) + I_imm6) & 0xFFFF);
    SET_CC ();
});

DEF_INST (LEA, FMT_RL, 0xF000, 0xE000, FLG_NONE, {
    RECORD_REG_DELTA (0, I_DR, REG (I_DR));
    RECORD_REG_DELTA (1, R_PSR, REG (R_PSR));
        
    REG (I_DR) = (REG (R_PC) + I_imm9) & 0xFFFF;
    SET_CC ();
});

DEF_INST (NOT, FMT_RR, 0xF03F, 0x903F, FLG_NONE, {
    RECORD_REG_DELTA (0, I_DR, REG (I_DR));
    RECORD_REG_DELTA (1, R_PSR, REG (R_PSR));
        
    REG (I_DR) = (REG (I_SR1) ^ 0xFFFF);
    SET_CC ();
}); 

/* RTI */
DEF_P_OP (RTI, FMT_, 0xFFFF, 0x8000);
/* Illegal without privilege mode, so left out...caught by illegal 
   instruction detection for now. */

DEF_INST (ST, FMT_RL, 0xF000, 0x3000, FLG_NONE, {
    RECORD_MEM_DELTA (0, ((REG (R_PC) + I_imm9) & 0xFFFF));

    write_memory ((REG (R_PC) + I_imm9) & 0xFFFF, REG (I_SR));
});

DEF_INST (STI, FMT_RL, 0xF000, 0xB000, FLG_NONE, {
    RECORD_MEM_DELTA (0, read_memory ((REG (R_PC) + I_imm9) & 0xFFFF));

    write_memory (read_memory ((REG (R_PC) + I_imm9) & 0xFFFF), REG (I_SR));
});

DEF_INST (STR, FMT_RRI6, 0xF000, 0x7000, FLG_NONE, {
    RECORD_MEM_DELTA (0, ((REG (I_BaseR) + I_imm6) & 0xFFFF));

    write_memory ((REG (I_BaseR) + I_imm6) & 0xFFFF, REG (I_SR));
});

DEF_P_OP (GETC,  FMT_, 0xFFFF, 0xF020);
DEF_P_OP (OUT,   FMT_, 0xFFFF, 0xF021);
DEF_P_OP (PUTS,  FMT_, 0xFFFF, 0xF022);
DEF_P_OP (IN,    FMT_, 0xFFFF, 0xF023);
DEF_P_OP (PUTSP, FMT_, 0xFFFF, 0xF024);
DEF_P_OP (HALT,  FMT_, 0xFFFF, 0xF025);

DEF_INST (TRAP, FMT_V, 0xFF00, 0xF000, FLG_SUBROUTINE, {
    RECORD_REG_DELTA (0, R_R7, REG (R_R7));
    
    REG (R_R7) = REG (R_PC);
    REG (R_PC) = read_memory (I_vec8);
});

/* for anything else, assume that it's data... */
DEF_P_OP (.FILL, FMT_16, 0x0000, 0x0000);

/* Undefine the field access macros. */
#undef INST
#undef I_DR
#undef I_SR1
#undef I_SR2
#undef I_BaseR
#undef I_SR
#undef I_CC
#undef I_vec8
#undef I_imm5
#undef I_imm6
#undef I_imm9
#undef I_imm11

/* Undefine operation macro. */
#undef SET_CC
EOF
cat>lc3.f<<EOF
/*									tab:8
 *
 * lc3.f - lexer for the LC-3 assembler
 *
 * "Copyright (c) 2003 by Steven S. Lumetta."
 *
 * Permission to use, copy, modify, and distribute this software and its
 * documentation for any purpose, without fee, and without written 
 * agreement is hereby granted, provided that the above copyright notice
 * and the following two paragraphs appear in all copies of this software,
 * that the files COPYING and NO_WARRANTY are included verbatim with
 * any distribution, and that the contents of the file README are included
 * verbatim as part of a file named README with any distribution.
 * 
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE TO ANY PARTY FOR DIRECT, 
 * INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT 
 * OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE AUTHOR 
 * HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 * THE AUTHOR SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT 
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR 
 * A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED HEREUNDER IS ON AN "AS IS" 
 * BASIS, AND THE AUTHOR NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, 
 * UPDATES, ENHANCEMENTS, OR MODIFICATIONS."
 *
 * Author:	    Steve Lumetta
 * Version:	    1
 * Creation Date:   18 October 2003
 * Filename:	    lc3.f
 * History:
 *	SSL	1	18 October 2003
 *		Copyright notices and Gnu Public License marker added.
 */

%option noyywrap nounput

%{

/* questions...

should the assembler allow colons after label names?  are the colons
part of the label?  Currently I allow only alpha followed by alphanum and _.

*/

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "symbol.h"

typedef enum opcode_t opcode_t;
enum opcode_t {
    /* no opcode seen (yet) */
    OP_NONE,

    /* real instruction opcodes */
    OP_ADD, OP_AND, OP_BR, OP_JMP, OP_JSR, OP_JSRR, OP_LD, OP_LDI, OP_LDR,
    OP_LEA, OP_NOT, OP_RTI, OP_ST, OP_STI, OP_STR, OP_TRAP,

    /* trap pseudo-ops */
    OP_GETC, OP_HALT, OP_IN, OP_OUT, OP_PUTS, OP_PUTSP,

    /* non-trap pseudo-ops */
    OP_FILL, OP_RET, OP_STRINGZ,

    /* directives */
    OP_BLKW, OP_END, OP_ORIG, 

    NUM_OPS
};

static const char* const opnames[NUM_OPS] = {
    /* no opcode seen (yet) */
    "missing opcode",

    /* real instruction opcodes */
    "ADD", "AND", "BR", "JMP", "JSR", "JSRR", "LD", "LDI", "LDR", "LEA",
    "NOT", "RTI", "ST", "STI", "STR", "TRAP",

    /* trap pseudo-ops */
    "GETC", "HALT", "IN", "OUT", "PUTS", "PUTSP",

    /* non-trap pseudo-ops */
    ".FILL", "RET", ".STRINGZ",

    /* directives */
    ".BLKW", ".END", ".ORIG",
};

typedef enum ccode_t ccode_t;
enum ccode_t {
    CC_    = 0,
    CC_P   = 0x0200,
    CC_Z   = 0x0400,
    CC_N   = 0x0800
};

typedef enum operands_t operands_t;
enum operands_t {
    O_RRR, O_RRI,
    O_RR,  O_RI,  O_RL,
    O_R,   O_I,   O_L,   O_S,
    O_,
    NUM_OPERANDS
};

static const int op_format_ok[NUM_OPS] = {
    /* no opcode seen (yet) */
    0x200, /* no opcode, no operands       */

    /* real instruction formats */
    0x003, /* ADD: RRR or RRI formats only */
    0x003, /* AND: RRR or RRI formats only */
    0x0C0, /* BR: I or L formats only      */
    0x020, /* JMP: R format only           */
    0x0C0, /* JSR: I or L formats only     */
    0x020, /* JSRR: R format only          */
    0x018, /* LD: RI or RL formats only    */
    0x018, /* LDI: RI or RL formats only   */
    0x002, /* LDR: RRI format only         */
    0x018, /* LEA: RI or RL formats only   */
    0x004, /* NOT: RR format only          */
    0x200, /* RTI: no operands allowed     */
    0x018, /* ST: RI or RL formats only    */
    0x018, /* STI: RI or RL formats only   */
    0x002, /* STR: RRI format only         */
    0x040, /* TRAP: I format only          */

    /* trap pseudo-op formats (no operands) */
    0x200, /* GETC: no operands allowed    */
    0x200, /* HALT: no operands allowed    */
    0x200, /* IN: no operands allowed      */
    0x200, /* OUT: no operands allowed     */
    0x200, /* PUTS: no operands allowed    */
    0x200, /* PUTSP: no operands allowed   */

    /* non-trap pseudo-op formats */
    0x0C0, /* .FILL: I or L formats only   */
    0x200, /* RET: no operands allowed     */
    0x100, /* .STRINGZ: S format only      */

    /* directive formats */
    0x040, /* .BLKW: I format only         */
    0x200, /* .END: no operands allowed    */
    0x040  /* .ORIG: I format only         */
};

typedef enum pre_parse_t pre_parse_t;
enum pre_parse_t {
    NO_PP =  0,
    PP_R1 =  1,
    PP_R2 =  2,
    PP_R3 =  4,
    PP_I2 =  8,
    PP_L2 = 16
};

static const pre_parse_t pre_parse[NUM_OPERANDS] = {
    (PP_R1 | PP_R2 | PP_R3), /* O_RRR */
    (PP_R1 | PP_R2),         /* O_RRI */
    (PP_R1 | PP_R2),         /* O_RR  */
    (PP_R1 | PP_I2),         /* O_RI  */
    (PP_R1 | PP_L2),         /* O_RL  */
    PP_R1,                   /* O_R   */
    NO_PP,                   /* O_I   */
    NO_PP,                   /* O_L   */
    NO_PP,                   /* O_S   */
    NO_PP                    /* O_    */
};

typedef struct inst_t inst_t;
struct inst_t {
    opcode_t op;
    ccode_t  ccode;
};

static int pass, line_num, num_errors, saw_orig, code_loc, saw_end;
static inst_t inst;
static FILE* symout;
static FILE* objout;

static void new_inst_line ();
static void bad_operands ();
static void unterminated_string ();
static void bad_line ();
static void line_ignored ();
static void parse_ccode (const char*);
static void generate_instruction (operands_t, const char*);
static void found_label (const char* lname);

%}

/* condition code specification */
CCODE    [Nn]?[Zz]?[Pp]?

/* operand types */
REGISTER [rR][0-7]
HEX      [xX][-]?[0-9a-fA-F]+
DECIMAL  [#]?[-]?[0-9]+
IMMED    {HEX}|{DECIMAL}
LABEL    [A-Za-z][A-Za-z_0-9]*
STRING   \\"([^\\"]*|(\\\\\\"))*\\"
UTSTRING \\"[^\\n\\r]*

/* operand and white space specification */
SPACE     [ \\t]
OP_SEP    {SPACE}*,{SPACE}*
COMMENT   [;][^\\n\\r]*
EMPTYLINE {SPACE}*{COMMENT}?
ENDLINE   {EMPTYLINE}\\r?\\n\\r?

/* operand formats */
O_RRR  {SPACE}+{REGISTER}{OP_SEP}{REGISTER}{OP_SEP}{REGISTER}{ENDLINE}
O_RRI  {SPACE}+{REGISTER}{OP_SEP}{REGISTER}{OP_SEP}{IMMED}{ENDLINE}
O_RR   {SPACE}+{REGISTER}{OP_SEP}{REGISTER}{ENDLINE}
O_RI   {SPACE}+{REGISTER}{OP_SEP}{IMMED}{ENDLINE}
O_RL   {SPACE}+{REGISTER}{OP_SEP}{LABEL}{ENDLINE}
O_R    {SPACE}+{REGISTER}{ENDLINE}
O_I    {SPACE}+{IMMED}{ENDLINE}
O_L    {SPACE}+{LABEL}{ENDLINE}
O_S    {SPACE}+{STRING}{ENDLINE}
O_UTS  {SPACE}+{UTSTRING}{ENDLINE}
O_     {ENDLINE}

/* need to define YY_INPUT... */

/* exclusive lexing states to read operands, eat garbage lines, and
   check for extra text after .END directive */
%x ls_operands ls_garbage ls_finished

%%

    /* rules for real instruction opcodes */
ADD       {inst.op = OP_ADD;   BEGIN (ls_operands);}
AND       {inst.op = OP_AND;   BEGIN (ls_operands);}
BR{CCODE} {inst.op = OP_BR;    parse_ccode (yytext + 2); BEGIN (ls_operands);}
JMP       {inst.op = OP_JMP;   BEGIN (ls_operands);}
JSRR      {inst.op = OP_JSRR;  BEGIN (ls_operands);}
JSR       {inst.op = OP_JSR;   BEGIN (ls_operands);}
LDI       {inst.op = OP_LDI;   BEGIN (ls_operands);}
LDR       {inst.op = OP_LDR;   BEGIN (ls_operands);}
LD        {inst.op = OP_LD;    BEGIN (ls_operands);}
LEA       {inst.op = OP_LEA;   BEGIN (ls_operands);}
NOT       {inst.op = OP_NOT;   BEGIN (ls_operands);}
RTI       {inst.op = OP_RTI;   BEGIN (ls_operands);}
STI       {inst.op = OP_STI;   BEGIN (ls_operands);}
STR       {inst.op = OP_STR;   BEGIN (ls_operands);}
ST        {inst.op = OP_ST;    BEGIN (ls_operands);}
TRAP      {inst.op = OP_TRAP;  BEGIN (ls_operands);}

    /* rules for trap pseudo-ols */
GETC      {inst.op = OP_GETC;  BEGIN (ls_operands);}
HALT      {inst.op = OP_HALT;  BEGIN (ls_operands);}
IN        {inst.op = OP_IN;    BEGIN (ls_operands);}
OUT       {inst.op = OP_OUT;   BEGIN (ls_operands);}
PUTS      {inst.op = OP_PUTS;  BEGIN (ls_operands);}
PUTSP     {inst.op = OP_PUTSP; BEGIN (ls_operands);}

    /* rules for non-trap pseudo-ops */
\\.FILL    {inst.op = OP_FILL;  BEGIN (ls_operands);}
RET       {inst.op = OP_RET;   BEGIN (ls_operands);}
\\.STRINGZ {inst.op = OP_STRINGZ; BEGIN (ls_operands);}

    /* rules for directives */
\\.BLKW    {inst.op = OP_BLKW; BEGIN (ls_operands);}
\\.END     {saw_end = 1;       BEGIN (ls_finished);}
\\.ORIG    {inst.op = OP_ORIG; BEGIN (ls_operands);}

    /* rules for operand formats */
<ls_operands>{O_RRR} {generate_instruction (O_RRR, yytext); BEGIN (0);}
<ls_operands>{O_RRI} {generate_instruction (O_RRI, yytext); BEGIN (0);}
<ls_operands>{O_RR}  {generate_instruction (O_RR, yytext);  BEGIN (0);}
<ls_operands>{O_RI}  {generate_instruction (O_RI, yytext);  BEGIN (0);}
<ls_operands>{O_RL}  {generate_instruction (O_RL, yytext);  BEGIN (0);}
<ls_operands>{O_R}   {generate_instruction (O_R, yytext);   BEGIN (0);}
<ls_operands>{O_I}   {generate_instruction (O_I, yytext);   BEGIN (0);}
<ls_operands>{O_L}   {generate_instruction (O_L, yytext);   BEGIN (0);}
<ls_operands>{O_S}   {generate_instruction (O_S, yytext);   BEGIN (0);}
<ls_operands>{O_}    {generate_instruction (O_, yytext);    BEGIN (0);}

    /* eat excess white space */
{SPACE}+ {}  
{ENDLINE} {new_inst_line (); /* a blank line */ }

    /* labels, with or without subsequent colons */\\
    /* 
       the colon form is used in some examples in the second edition
       of the book, but may be removed in the third; it also allows 
       labels to use opcode and pseudo-op names, etc., however.
     */
{LABEL}          {found_label (yytext);}
{LABEL}{SPACE}*: {found_label (yytext);}

    /* error handling??? */
<ls_operands>{O_UTS} {unterminated_string (); BEGIN (0);}
<ls_operands>[^\\n\\r]*{ENDLINE} {bad_operands (); BEGIN (0);}
{O_RRR}|{O_RRI}|{O_RR}|{O_RI}|{O_RL}|{O_R}|{O_I}|{O_S}|{O_UTS} {
    bad_operands ();
}

. {BEGIN (ls_garbage);}
<ls_garbage>[^\\n\\r]*{ENDLINE} {bad_line (); BEGIN (0);}

    /* parsing after the .END directive */
<ls_finished>{ENDLINE}|{EMPTYLINE}     {new_inst_line (); /* a blank line  */}
<ls_finished>.*({ENDLINE}|{EMPTYLINE}) {line_ignored (); return 0;}

%%

int
main (int argc, char** argv)
{
    int len;
    char* ext;
    char* fname;

    if (argc != 2) {
        fprintf (stderr, "usage: %s <ASM filename>\\n", argv[0]);
	return 1;
    }

    /* Make our own copy of the filename. */
    len = strlen (argv[1]);
    if ((fname = malloc (len + 5)) == NULL) {
        perror ("malloc");
	return 3;
    }
    strcpy (fname, argv[1]);

    /* Check for .asm extension; if not found, add it. */
    if ((ext = strrchr (fname, '.')) == NULL || strcmp (ext, ".asm") != 0) {
	ext = fname + len;
        strcpy (ext, ".asm");
    }

    /* Open input file. */
    if ((lc3in = fopen (fname, "r")) == NULL) {
        fprintf (stderr, "Could not open %s for reading.\\n", fname);
	return 2;
    }

    /* Open output files. */
    strcpy (ext, ".obj");
    if ((objout = fopen (fname, "w")) == NULL) {
        fprintf (stderr, "Could not open %s for writing.\\n", fname);
	return 2;
    }
    strcpy (ext, ".sym");
    if ((symout = fopen (fname, "w")) == NULL) {
        fprintf (stderr, "Could not open %s for writing.\\n", fname);
	return 2;
    }
    /* FIXME: Do we really need to exactly match old format for compatibility 
       with Windows simulator? */
    fprintf (symout, "// Symbol table\\n");
    fprintf (symout, "// Scope level 0:\\n");
    fprintf (symout, "//\\tSymbol Name       Page Address\\n");
    fprintf (symout, "//\\t----------------  ------------\\n");

    puts ("STARTING PASS 1");
    pass = 1;
    line_num = 0;
    num_errors = 0;
    saw_orig = 0;
    code_loc = 0x3000;
    saw_end = 0;
    new_inst_line ();
    yylex ();
    if (saw_orig == 0) {
        if (num_errors == 0 && !saw_end)
	    fprintf (stderr, "%3d: file contains only comments\\n", line_num);
        else {
	    if (saw_end == 0)
		fprintf (stderr, "%3d: no .ORIG or .END directive found\\n", 
			 line_num);
	    else
		fprintf (stderr, "%3d: no .ORIG directive found\\n", line_num);
	}
	num_errors++;
    } else if (saw_end == 0 ) {
	fprintf (stderr, "%3d: no .END directive found\\n", line_num);
	num_errors++;
    }
    printf ("%d errors found in first pass.\\n", num_errors);
    if (num_errors > 0)
    	return 1;
    if (fseek (lc3in, 0, SEEK_SET) != 0) {
        perror ("fseek to start of ASM file");
	return 3;
    }
    yyrestart (lc3in);
    /* Return lexer to initial state.  It is otherwise left in ls_finished
       if an .END directive was seen. */
    BEGIN (0);

    puts ("STARTING PASS 2");
    pass = 2;
    line_num = 0;
    num_errors = 0;
    saw_orig = 0;
    code_loc = 0x3000;
    saw_end = 0;
    new_inst_line ();
    yylex ();
    printf ("%d errors found in second pass.\\n", num_errors);
    if (num_errors > 0)
    	return 1;

    fprintf (symout, "\\n");
    fclose (symout);
    fclose (objout);

    return 0;
}

static void
new_inst_line () 
{
    inst.op = OP_NONE;
    inst.ccode = CC_;
    line_num++;
}

static void
bad_operands ()
{
    fprintf (stderr, "%3d: illegal operands for %s\\n",
	     line_num, opnames[inst.op]);
    num_errors++;
    new_inst_line ();
}

static void
unterminated_string ()
{
    fprintf (stderr, "%3d: unterminated string\\n", line_num);
    num_errors++;
    new_inst_line ();
}

static void 
bad_line ()
{
    fprintf (stderr, "%3d: contains unrecognizable characters\\n",
	     line_num);
    num_errors++;
    new_inst_line ();
}

static void 
line_ignored ()
{
    if (pass == 1)
	fprintf (stderr, "%3d: WARNING: all text after .END ignored\\n",
		 line_num);
}

static int
read_val (const char* s, int* vptr, int bits)
{
    char* trash;
    long v;

    if (*s == 'x' || *s == 'X')
	v = strtol (s + 1, &trash, 16);
    else {
	if (*s == '#')
	    s++;
	v = strtol (s, &trash, 10);
    }
    if (0x10000 > v && 0x8000 <= v)
        v |= -65536L;   /* handles 64-bit longs properly */
    if (v < -(1L << (bits - 1)) || v >= (1L << bits)) {
	fprintf (stderr, "%3d: constant outside of allowed range\\n", line_num);
	num_errors++;
	return -1;
    }
    if ((v & (1UL << (bits - 1))) != 0)
	v |= ~((1UL << bits) - 1);
    *vptr = v;
    return 0;
}

static void
write_value (int val)
{
    unsigned char out[2];

    code_loc = (code_loc + 1) & 0xFFFF;
    if (pass == 1)
        return;
    /* FIXME: just htons... */
    out[0] = (val >> 8);
    out[1] = (val & 0xFF);
    fwrite (out, 2, 1, objout);
}

static char*
sym_name (const char* name)
{
    char* local = strdup (name);
    char* cut;

    /* Not fast, but no limit on label length...who cares? */
    for (cut = local; 
         *cut != 0 && !isspace ((unsigned char)*cut) && *cut != ':'; cut++) { }
    *cut = 0;

    return local;
}

static int
find_label (const char* optarg, int bits)
{
    char* local;
    symbol_t* label;
    int limit, value;

    if (pass == 1)
        return 0;

    local = sym_name (optarg);
    label = find_symbol (local, NULL);
    if (label != NULL) {
	value = label->addr;
	if (bits != 16) { /* Everything except 16 bits is PC-relative. */
	    limit = (1L << (bits - 1));
	    value -= code_loc + 1;
	    if (value < -limit || value >= limit) {
	        fprintf (stderr, "%3d: label \\"%s\\" at distance %d (allowed "
			 "range is %d to %d)\\n", line_num, local, value,
			 -limit, limit - 1);
	        goto bad_label;
	    }
	    return value;
	}
	free (local);
        return label->addr;
    }
    fprintf (stderr, "%3d: unknown label \\"%s\\"\\n", line_num, local);

bad_label:
    num_errors++;
    free (local);
    return 0;
}

static void 
generate_instruction (operands_t operands, const char* opstr)
{
    int val, r1, r2, r3;
    const char* o1;
    const char* o2;
    const char* o3;
    const char* str;

    if ((op_format_ok[inst.op] & (1UL << operands)) == 0) {
	bad_operands ();
	return;
    }
    o1 = opstr;
    while (isspace ((unsigned char)*o1)) o1++;
    if ((o2 = strchr (o1, ((unsigned char)','))) != NULL) {
        o2++;
	while (isspace ((unsigned char)*o2)) o2++;
	if ((o3 = strchr (o2, ((unsigned char)','))) != NULL) {
	    o3++;
	    while (isspace (((unsigned char)*o3))) o3++;
	}
    } else
    	o3 = NULL;
    if (inst.op == OP_ORIG) {
	if (saw_orig == 0) {
	    if (read_val (o1, &code_loc, 16) == -1)
		/* Pick a value; the error prevents code generation. */
		code_loc = 0x3000; 
	    else {
	        write_value (code_loc);
		code_loc--; /* Starting point doesn't count as code. */
	    }
	    saw_orig = 1;
	} else if (saw_orig == 1) {
	    fprintf (stderr, "%3d: multiple .ORIG directives found\\n",
		     line_num);
	    saw_orig = 2;
	}
	new_inst_line ();
	return;
    }
    if (saw_orig == 0) {
	fprintf (stderr, "%3d: instruction appears before .ORIG\\n",
		 line_num);
	num_errors++;
	new_inst_line ();
	saw_orig = 2;
	return;
    }
    if ((pre_parse[operands] & PP_R1) != 0)
        r1 = o1[1] - '0';
    if ((pre_parse[operands] & PP_R2) != 0)
        r2 = o2[1] - '0';
    if ((pre_parse[operands] & PP_R3) != 0)
        r3 = o3[1] - '0';
    if ((pre_parse[operands] & PP_I2) != 0)
        (void)read_val (o2, &val, 9);
    if ((pre_parse[operands] & PP_L2) != 0)
        val = find_label (o2, 9);

    switch (inst.op) {
	/* Generate real instruction opcodes. */
	case OP_ADD:
	    if (operands == O_RRI) {
	    	/* Check or read immediate range (error in first pass
		   prevents execution of second, so never fails). */
	        (void)read_val (o3, &val, 5);
		write_value (0x1020 | (r1 << 9) | (r2 << 6) | (val & 0x1F));
	    } else
		write_value (0x1000 | (r1 << 9) | (r2 << 6) | r3);
	    break;
	case OP_AND:
	    if (operands == O_RRI) {
	    	/* Check or read immediate range (error in first pass
		   prevents execution of second, so never fails). */
	        (void)read_val (o3, &val, 5);
		write_value (0x5020 | (r1 << 9) | (r2 << 6) | (val & 0x1F));
	    } else
		write_value (0x5000 | (r1 << 9) | (r2 << 6) | r3);
	    break;
	case OP_BR:
	    if (operands == O_I)
	        (void)read_val (o1, &val, 9);
	    else /* O_L */
	        val = find_label (o1, 9);
	    write_value (inst.ccode | (val & 0x1FF));
	    break;
	case OP_JMP:
	    write_value (0xC000 | (r1 << 6));
	    break;
	case OP_JSR:
	    if (operands == O_I)
	        (void)read_val (o1, &val, 11);
	    else /* O_L */
	        val = find_label (o1, 11);
	    write_value (0x4800 | (val & 0x7FF));
	    break;
	case OP_JSRR:
	    write_value (0x4000 | (r1 << 6));
	    break;
	case OP_LD:
	    write_value (0x2000 | (r1 << 9) | (val & 0x1FF));
	    break;
	case OP_LDI:
	    write_value (0xA000 | (r1 << 9) | (val & 0x1FF));
	    break;
	case OP_LDR:
	    (void)read_val (o3, &val, 6);
	    write_value (0x6000 | (r1 << 9) | (r2 << 6) | (val & 0x3F));
	    break;
	case OP_LEA:
	    write_value (0xE000 | (r1 << 9) | (val & 0x1FF));
	    break;
	case OP_NOT:
	    write_value (0x903F | (r1 << 9) | (r2 << 6));
	    break;
	case OP_RTI:
	    write_value (0x8000);
	    break;
	case OP_ST:
	    write_value (0x3000 | (r1 << 9) | (val & 0x1FF));
	    break;
	case OP_STI:
	    write_value (0xB000 | (r1 << 9) | (val & 0x1FF));
	    break;
	case OP_STR:
	    (void)read_val (o3, &val, 6);
	    write_value (0x7000 | (r1 << 9) | (r2 << 6) | (val & 0x3F));
	    break;
	case OP_TRAP:
	    (void)read_val (o1, &val, 8);
	    write_value (0xF000 | (val & 0xFF));
	    break;

	/* Generate trap pseudo-ops. */
	case OP_GETC:  write_value (0xF020); break;
	case OP_HALT:  write_value (0xF025); break;
	case OP_IN:    write_value (0xF023); break;
	case OP_OUT:   write_value (0xF021); break;
	case OP_PUTS:  write_value (0xF022); break;
	case OP_PUTSP: write_value (0xF024); break;

	/* Generate non-trap pseudo-ops. */
    	case OP_FILL:
	    if (operands == O_I) {
		(void)read_val (o1, &val, 16);
		val &= 0xFFFF;
	    } else /* O_L */
		val = find_label (o1, 16);
	    write_value (val);
    	    break;
	case OP_RET:   
	    write_value (0xC1C0); 
	    break;
	case OP_STRINGZ:
	    /* We must count locations written in pass 1;
	       write_value squashes the writes. */
	    for (str = o1 + 1; str[0] != '\\"'; str++) {
		if (str[0] == '\\\\') {
		    switch (str[1]) {
			case 'a': write_value ('\\a'); str++; break;
			case 'b': write_value ('\\b'); str++; break;
			case 'e': write_value ('\\e'); str++; break;
			case 'f': write_value ('\\f'); str++; break;
			case 'n': write_value ('\\n'); str++; break;
			case 'r': write_value ('\\r'); str++; break;
			case 't': write_value ('\\t'); str++; break;
			case 'v': write_value ('\\v'); str++; break;
			case '\\\\': write_value ('\\\\'); str++; break;
			case '\\"': write_value ('\\"'); str++; break;
			/* FIXME: support others too? */
			default: write_value (str[1]); str++; break;
		    }
		} else {
		    if (str[0] == '\\n')
		        line_num++;
		    write_value (*str);
		}
	    }
	    write_value (0);
	    break;
	case OP_BLKW:
	    (void)read_val (o1, &val, 16);
	    val &= 0xFFFF;
	    while (val-- > 0)
	        write_value (0x0000);
	    break;
	
	/* Handled earlier or never used, so never seen here. */
	case OP_NONE:
        case OP_ORIG:
        case OP_END:
	case NUM_OPS:
	    break;
    }
    new_inst_line ();
}

static void 
parse_ccode (const char* ccstr)
{
    if (*ccstr == 'N' || *ccstr == 'n') {
	inst.ccode |= CC_N;
        ccstr++;
    }
    if (*ccstr == 'Z' || *ccstr == 'z') {
	inst.ccode |= CC_Z;
        ccstr++;
    }
    if (*ccstr == 'P' || *ccstr == 'p')
	inst.ccode |= CC_P;

    /* special case: map BR to BRnzp */
    if (inst.ccode == CC_)
        inst.ccode = CC_P | CC_Z | CC_N;
}

static void
found_label (const char* lname) 
{
    char* local = sym_name (lname);

    if (pass == 1) {
	if (saw_orig == 0) {
	    fprintf (stderr, "%3d: label appears before .ORIG\\n", line_num);
	    num_errors++;
	} else if (add_symbol (local, code_loc, 0) == -1) {
	    fprintf (stderr, "%3d: label %s has already appeared\\n", 
	    	     line_num, local);
	    num_errors++;
	} else
	    fprintf (symout, "//\\t%-16s  %04X\\n", local, code_loc);
    }

    free (local);
}

EOF
cat>lc3convert.f<<EOF
/*									tab:8
 *
 * lc3convert.f - LC-3 binary and hexadecimal file conversion tool
 *
 * "Copyright (c) 2003 by Steven S. Lumetta."
 *
 * Permission to use, copy, modify, and distribute this software and its
 * documentation for any purpose, without fee, and without written 
 * agreement is hereby granted, provided that the above copyright notice
 * and the following two paragraphs appear in all copies of this software,
 * that the files COPYING and NO_WARRANTY are included verbatim with
 * any distribution, and that the contents of the file README are included
 * verbatim as part of a file named README with any distribution.
 * 
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE TO ANY PARTY FOR DIRECT, 
 * INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT 
 * OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE AUTHOR 
 * HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 * THE AUTHOR SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT 
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR 
 * A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED HEREUNDER IS ON AN "AS IS" 
 * BASIS, AND THE AUTHOR NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, 
 * UPDATES, ENHANCEMENTS, OR MODIFICATIONS."
 *
 * Author:	    Steve Lumetta
 * Version:	    2
 * Creation Date:   30 October 2003
 * Filename:	    lc3convert.f
 * History:
 *	SSL	2	31 October 2003
 *		Finished initial version.
 *	SSL	1	30 October 2003
 *		Started paring down lc3as code.
 */

%option noyywrap nounput

%{

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>


static int line_num, num_errors, parse_hex, binary_value, bin_count, hex_count;
static FILE* objout;

static void new_inst_line ();
static void bad_line ();
static void generate_bin_instruction (int value);
static void generate_hex_instruction (const char* val_str);
static void end_current_bin_line ();

%}

/* 
   Operand types--note that hexadecimal does not require the "x" prefix
   for conversion.
*/
HEX      [xX]?[-]?[0-9a-fA-F]+
BINARY   [01]

/* comment and white space specification */
SPACE    [ \\t]
COMMENT  [;][^\\n\\r]*
ENDLINE  {SPACE}*{COMMENT}?\\r?\\n\\r?

/* exclusive lexing states to read binary and hexadecimal files */
%x ls_binary ls_hexadecimal ls_bin_garbage ls_hex_garbage

%%
    /* Choose appropriate parser. */
    if (parse_hex) {
        BEGIN (ls_hexadecimal);
    } else {
        BEGIN (ls_binary);
    }

    /* binary parser */
<ls_binary>{BINARY} {
    if (++bin_count == 17) {
        fprintf (stderr, "%3d: line contains more than 16 digits\\n", line_num);
	num_errors++;
    } else {
        binary_value <<= 1;
	if (*yytext == '1')
	    binary_value++;
    }
}
<ls_binary>{ENDLINE} {end_current_bin_line ();}
<ls_binary><<EOF>> {end_current_bin_line (); return 0;}

    /* hexadecimal parser */
<ls_hexadecimal>{HEX} {generate_hex_instruction (yytext);}
<ls_hexadecimal>{ENDLINE} {new_inst_line (); /* a blank line */ }


    /* eat excess white space (both parsers) */
<ls_binary,ls_hexadecimal>{SPACE}+ {}  


    /* error handling (replicated because of substate use) */

<ls_binary>. {BEGIN (ls_bin_garbage);}
<ls_bin_garbage>[^\\n\\r]*{ENDLINE} {bad_line (); BEGIN (ls_binary);}
<ls_bin_garbage><<EOF>> {bad_line (); BEGIN (ls_binary); return 0;}

<ls_hexadecimal>. {BEGIN (ls_hex_garbage);}
<ls_hex_garbage>[^\\n\\r]*{ENDLINE} {bad_line (); BEGIN (ls_hexadecimal);}
<ls_hex_garbage><<EOF>> {bad_line (); BEGIN (ls_hexadecimal); return 0;}

%%

int
main (int argc, char** argv)
{
    int len, parse_error = 0;
    char* use_ext;
    char* ext;
    char* fname;

    if (argc == 3) {
	if (strcmp (argv[1], "-b2") == 0)
	    parse_hex = 0;
	else if (strcmp (argv[1], "-b16") == 0)
	    parse_hex = 1;
	else
	    parse_error = 1;
    } else
	parse_hex = 0;

    if (parse_error || argc < 2 || argc > 3) {
        fprintf (stderr, "usage: %s [-b2] <BIN filename>\\n", argv[0]);
        fprintf (stderr, "       %s -b16 <HEX filename>\\n", argv[0]);
	return 1;
    }

    /* Make our own copy of the filename. */
    len = strlen (argv[argc - 1]);
    if ((fname = malloc (len + 5)) == NULL) {
        perror ("malloc");
	return 3;
    }
    strcpy (fname, argv[argc - 1]);

    /* Check for .bin or .hex extension; if not found, add it. */
    use_ext = (parse_hex ? ".hex" : ".bin");
    if ((ext = strrchr (fname, '.')) == NULL || strcmp (ext, use_ext) != 0) {
	ext = fname + len;
        strcpy (ext, use_ext);
    }

    /* Open input file. */
    if ((lc3convertin = fopen (fname, "r")) == NULL) {
        fprintf (stderr, "Could not open %s for reading.\\n", fname);
	return 2;
    }

    /* Open output files. */
    strcpy (ext, ".obj");
    if ((objout = fopen (fname, "w")) == NULL) {
        fprintf (stderr, "Could not open %s for writing.\\n", fname);
	return 2;
    }

    line_num = 0;
    num_errors = 0;
    new_inst_line ();
    yylex ();
    printf ("%d errors found.\\n", num_errors);
    if (num_errors > 0)
    	return 1;

    fclose (objout);

    return 0;
}

static void
new_inst_line () 
{
    binary_value = 0;
    bin_count = 0;
    hex_count = 0;
    line_num++;
}

static void 
bad_line ()
{
    fprintf (stderr, "%3d: contains unrecognizable characters\\n",
	     line_num);
    num_errors++;
    new_inst_line ();
}

static int
read_val (const char* s, int* vptr, int bits)
{
    char* trash;
    long v;

    if (*s == 'x' || *s == 'X')
	s++;
    v = strtol (s, &trash, 16);
    if (0x10000 > v && 0x8000 <= v)
        v |= -65536L;   /* handles 64-bit longs properly */
    if (v < -(1L << (bits - 1)) || v >= (1L << bits)) {
	fprintf (stderr, "%3d: constant outside of allowed range\\n", line_num);
	num_errors++;
	return -1;
    }
    if ((v & (1UL << (bits - 1))) != 0)
	v |= ~((1UL << bits) - 1);
    *vptr = v;
    return 0;
}

static void
write_value (int val)
{
    unsigned char out[2];

    /* FIXME: just htons... */
    out[0] = (val >> 8);
    out[1] = (val & 0xFF);
    fwrite (out, 2, 1, objout);
}

static void 
generate_bin_instruction (int value)
{
    write_value (value);
    new_inst_line ();
}

static void 
generate_hex_instruction (const char* val_str)
{
    int value;

    if (0 == hex_count) {
	if (0 == read_val (val_str, &value, 16)) {
	    write_value (value);
	}
	hex_count = 1;
    } else {
        fprintf (stderr, "%3d: line contains multiple hex values\\n", line_num);
	num_errors++;
    }
}

static void
end_current_bin_line () 
{
    if (bin_count == 0) { 
        /* a blank line */
	new_inst_line ();
    } else {
	if (bin_count < 16) {
	    fprintf (stderr, "%3d: line contains only %d digits\\n", line_num,
		     bin_count);
	    num_errors++;
	}
	generate_bin_instruction (binary_value);
    }
}

EOF
cat>lc3os.asm<<EOF
;##############################################################################
;#
;# lc3os.asm -- the LC-3 operating system
;#
;#  "Copyright (c) 2003 by Steven S. Lumetta."
;# 
;#  Permission to use, copy, modify, and distribute this software and its
;#  documentation for any purpose, without fee, and without written 
;#  agreement is hereby granted, provided that the above copyright notice
;#  and the following two paragraphs appear in all copies of this software,
;#  that the files COPYING and NO_WARRANTY are included verbatim with
;#  any distribution, and that the contents of the file README are included
;#  verbatim as part of a file named README with any distribution.
;#  
;#  IN NO EVENT SHALL THE AUTHOR BE LIABLE TO ANY PARTY FOR DIRECT, 
;#  INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT 
;#  OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE AUTHOR 
;#  HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
;#  
;#  THE AUTHOR SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT 
;#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR 
;#  A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED HEREUNDER IS ON AN "AS IS" 
;#  BASIS, AND THE AUTHOR NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, 
;#  UPDATES, ENHANCEMENTS, OR MODIFICATIONS."
;#
;#  Author:		Steve Lumetta
;#  Version:		1
;#  Creation Date:	18 October 2003
;#  Filename:		lc3os.asm
;#  History:		
;# 	SSL	1	18 October 2003
;# 		Copyright notices and Gnu Public License marker added.
;#
;##############################################################################

	.ORIG x0000

; the TRAP vector table
	.FILL BAD_TRAP	; x00
	.FILL BAD_TRAP	; x01
	.FILL BAD_TRAP	; x02
	.FILL BAD_TRAP	; x03
	.FILL BAD_TRAP	; x04
	.FILL BAD_TRAP	; x05
	.FILL BAD_TRAP	; x06
	.FILL BAD_TRAP	; x07
	.FILL BAD_TRAP	; x08
	.FILL BAD_TRAP	; x09
	.FILL BAD_TRAP	; x0A
	.FILL BAD_TRAP	; x0B
	.FILL BAD_TRAP	; x0C
	.FILL BAD_TRAP	; x0D
	.FILL BAD_TRAP	; x0E
	.FILL BAD_TRAP	; x0F
	.FILL BAD_TRAP	; x10
	.FILL BAD_TRAP	; x11
	.FILL BAD_TRAP	; x12
	.FILL BAD_TRAP	; x13
	.FILL BAD_TRAP	; x14
	.FILL BAD_TRAP	; x15
	.FILL BAD_TRAP	; x16
	.FILL BAD_TRAP	; x17
	.FILL BAD_TRAP	; x18
	.FILL BAD_TRAP	; x19
	.FILL BAD_TRAP	; x1A
	.FILL BAD_TRAP	; x1B
	.FILL BAD_TRAP	; x1C
	.FILL BAD_TRAP	; x1D
	.FILL BAD_TRAP	; x1E
	.FILL BAD_TRAP	; x1F
	.FILL TRAP_GETC	; x20
	.FILL TRAP_OUT	; x21
	.FILL TRAP_PUTS	; x22
	.FILL TRAP_IN	; x23
	.FILL TRAP_PUTSP ; x24
	.FILL TRAP_HALT	; x25
	.FILL BAD_TRAP	; x26
	.FILL BAD_TRAP	; x27
	.FILL BAD_TRAP	; x28
	.FILL BAD_TRAP	; x29
	.FILL BAD_TRAP	; x2A
	.FILL BAD_TRAP	; x2B
	.FILL BAD_TRAP	; x2C
	.FILL BAD_TRAP	; x2D
	.FILL BAD_TRAP	; x2E
	.FILL BAD_TRAP	; x2F
	.FILL BAD_TRAP	; x30
	.FILL BAD_TRAP	; x31
	.FILL BAD_TRAP	; x32
	.FILL BAD_TRAP	; x33
	.FILL BAD_TRAP	; x34
	.FILL BAD_TRAP	; x35
	.FILL BAD_TRAP	; x36
	.FILL BAD_TRAP	; x37
	.FILL BAD_TRAP	; x38
	.FILL BAD_TRAP	; x39
	.FILL BAD_TRAP	; x3A
	.FILL BAD_TRAP	; x3B
	.FILL BAD_TRAP	; x3C
	.FILL BAD_TRAP	; x3D
	.FILL BAD_TRAP	; x3E
	.FILL BAD_TRAP	; x3F
	.FILL BAD_TRAP	; x40
	.FILL BAD_TRAP	; x41
	.FILL BAD_TRAP	; x42
	.FILL BAD_TRAP	; x43
	.FILL BAD_TRAP	; x44
	.FILL BAD_TRAP	; x45
	.FILL BAD_TRAP	; x46
	.FILL BAD_TRAP	; x47
	.FILL BAD_TRAP	; x48
	.FILL BAD_TRAP	; x49
	.FILL BAD_TRAP	; x4A
	.FILL BAD_TRAP	; x4B
	.FILL BAD_TRAP	; x4C
	.FILL BAD_TRAP	; x4D
	.FILL BAD_TRAP	; x4E
	.FILL BAD_TRAP	; x4F
	.FILL BAD_TRAP	; x50
	.FILL BAD_TRAP	; x51
	.FILL BAD_TRAP	; x52
	.FILL BAD_TRAP	; x53
	.FILL BAD_TRAP	; x54
	.FILL BAD_TRAP	; x55
	.FILL BAD_TRAP	; x56
	.FILL BAD_TRAP	; x57
	.FILL BAD_TRAP	; x58
	.FILL BAD_TRAP	; x59
	.FILL BAD_TRAP	; x5A
	.FILL BAD_TRAP	; x5B
	.FILL BAD_TRAP	; x5C
	.FILL BAD_TRAP	; x5D
	.FILL BAD_TRAP	; x5E
	.FILL BAD_TRAP	; x5F
	.FILL BAD_TRAP	; x60
	.FILL BAD_TRAP	; x61
	.FILL BAD_TRAP	; x62
	.FILL BAD_TRAP	; x63
	.FILL BAD_TRAP	; x64
	.FILL BAD_TRAP	; x65
	.FILL BAD_TRAP	; x66
	.FILL BAD_TRAP	; x67
	.FILL BAD_TRAP	; x68
	.FILL BAD_TRAP	; x69
	.FILL BAD_TRAP	; x6A
	.FILL BAD_TRAP	; x6B
	.FILL BAD_TRAP	; x6C
	.FILL BAD_TRAP	; x6D
	.FILL BAD_TRAP	; x6E
	.FILL BAD_TRAP	; x6F
	.FILL BAD_TRAP	; x70
	.FILL BAD_TRAP	; x71
	.FILL BAD_TRAP	; x72
	.FILL BAD_TRAP	; x73
	.FILL BAD_TRAP	; x74
	.FILL BAD_TRAP	; x75
	.FILL BAD_TRAP	; x76
	.FILL BAD_TRAP	; x77
	.FILL BAD_TRAP	; x78
	.FILL BAD_TRAP	; x79
	.FILL BAD_TRAP	; x7A
	.FILL BAD_TRAP	; x7B
	.FILL BAD_TRAP	; x7C
	.FILL BAD_TRAP	; x7D
	.FILL BAD_TRAP	; x7E
	.FILL BAD_TRAP	; x7F
	.FILL BAD_TRAP	; x80
	.FILL BAD_TRAP	; x81
	.FILL BAD_TRAP	; x82
	.FILL BAD_TRAP	; x83
	.FILL BAD_TRAP	; x84
	.FILL BAD_TRAP	; x85
	.FILL BAD_TRAP	; x86
	.FILL BAD_TRAP	; x87
	.FILL BAD_TRAP	; x88
	.FILL BAD_TRAP	; x89
	.FILL BAD_TRAP	; x8A
	.FILL BAD_TRAP	; x8B
	.FILL BAD_TRAP	; x8C
	.FILL BAD_TRAP	; x8D
	.FILL BAD_TRAP	; x8E
	.FILL BAD_TRAP	; x8F
	.FILL BAD_TRAP	; x90
	.FILL BAD_TRAP	; x91
	.FILL BAD_TRAP	; x92
	.FILL BAD_TRAP	; x93
	.FILL BAD_TRAP	; x94
	.FILL BAD_TRAP	; x95
	.FILL BAD_TRAP	; x96
	.FILL BAD_TRAP	; x97
	.FILL BAD_TRAP	; x98
	.FILL BAD_TRAP	; x99
	.FILL BAD_TRAP	; x9A
	.FILL BAD_TRAP	; x9B
	.FILL BAD_TRAP	; x9C
	.FILL BAD_TRAP	; x9D
	.FILL BAD_TRAP	; x9E
	.FILL BAD_TRAP	; x9F
	.FILL BAD_TRAP	; xA0
	.FILL BAD_TRAP	; xA1
	.FILL BAD_TRAP	; xA2
	.FILL BAD_TRAP	; xA3
	.FILL BAD_TRAP	; xA4
	.FILL BAD_TRAP	; xA5
	.FILL BAD_TRAP	; xA6
	.FILL BAD_TRAP	; xA7
	.FILL BAD_TRAP	; xA8
	.FILL BAD_TRAP	; xA9
	.FILL BAD_TRAP	; xAA
	.FILL BAD_TRAP	; xAB
	.FILL BAD_TRAP	; xAC
	.FILL BAD_TRAP	; xAD
	.FILL BAD_TRAP	; xAE
	.FILL BAD_TRAP	; xAF
	.FILL BAD_TRAP	; xB0
	.FILL BAD_TRAP	; xB1
	.FILL BAD_TRAP	; xB2
	.FILL BAD_TRAP	; xB3
	.FILL BAD_TRAP	; xB4
	.FILL BAD_TRAP	; xB5
	.FILL BAD_TRAP	; xB6
	.FILL BAD_TRAP	; xB7
	.FILL BAD_TRAP	; xB8
	.FILL BAD_TRAP	; xB9
	.FILL BAD_TRAP	; xBA
	.FILL BAD_TRAP	; xBB
	.FILL BAD_TRAP	; xBC
	.FILL BAD_TRAP	; xBD
	.FILL BAD_TRAP	; xBE
	.FILL BAD_TRAP	; xBF
	.FILL BAD_TRAP	; xC0
	.FILL BAD_TRAP	; xC1
	.FILL BAD_TRAP	; xC2
	.FILL BAD_TRAP	; xC3
	.FILL BAD_TRAP	; xC4
	.FILL BAD_TRAP	; xC5
	.FILL BAD_TRAP	; xC6
	.FILL BAD_TRAP	; xC7
	.FILL BAD_TRAP	; xC8
	.FILL BAD_TRAP	; xC9
	.FILL BAD_TRAP	; xCA
	.FILL BAD_TRAP	; xCB
	.FILL BAD_TRAP	; xCC
	.FILL BAD_TRAP	; xCD
	.FILL BAD_TRAP	; xCE
	.FILL BAD_TRAP	; xCF
	.FILL BAD_TRAP	; xD0
	.FILL BAD_TRAP	; xD1
	.FILL BAD_TRAP	; xD2
	.FILL BAD_TRAP	; xD3
	.FILL BAD_TRAP	; xD4
	.FILL BAD_TRAP	; xD5
	.FILL BAD_TRAP	; xD6
	.FILL BAD_TRAP	; xD7
	.FILL BAD_TRAP	; xD8
	.FILL BAD_TRAP	; xD9
	.FILL BAD_TRAP	; xDA
	.FILL BAD_TRAP	; xDB
	.FILL BAD_TRAP	; xDC
	.FILL BAD_TRAP	; xDD
	.FILL BAD_TRAP	; xDE
	.FILL BAD_TRAP	; xDF
	.FILL BAD_TRAP	; xE0
	.FILL BAD_TRAP	; xE1
	.FILL BAD_TRAP	; xE2
	.FILL BAD_TRAP	; xE3
	.FILL BAD_TRAP	; xE4
	.FILL BAD_TRAP	; xE5
	.FILL BAD_TRAP	; xE6
	.FILL BAD_TRAP	; xE7
	.FILL BAD_TRAP	; xE8
	.FILL BAD_TRAP	; xE9
	.FILL BAD_TRAP	; xEA
	.FILL BAD_TRAP	; xEB
	.FILL BAD_TRAP	; xEC
	.FILL BAD_TRAP	; xED
	.FILL BAD_TRAP	; xEE
	.FILL BAD_TRAP	; xEF
	.FILL BAD_TRAP	; xF0
	.FILL BAD_TRAP	; xF1
	.FILL BAD_TRAP	; xF2
	.FILL BAD_TRAP	; xF3
	.FILL BAD_TRAP	; xF4
	.FILL BAD_TRAP	; xF5
	.FILL BAD_TRAP	; xF6
	.FILL BAD_TRAP	; xF7
	.FILL BAD_TRAP	; xF8
	.FILL BAD_TRAP	; xF9
	.FILL BAD_TRAP	; xFA
	.FILL BAD_TRAP	; xFB
	.FILL BAD_TRAP	; xFC
	.FILL BAD_TRAP	; xFD
	.FILL BAD_TRAP	; xFE
	.FILL BAD_TRAP	; xFF

; the interrupt vector table
	.FILL INT_PRIV	; x00
	.FILL INT_ILL	; x01
	.FILL BAD_INT	; x02
	.FILL BAD_INT	; x03
	.FILL BAD_INT	; x04
	.FILL BAD_INT	; x05
	.FILL BAD_INT	; x06
	.FILL BAD_INT	; x07
	.FILL BAD_INT	; x08
	.FILL BAD_INT	; x09
	.FILL BAD_INT	; x0A
	.FILL BAD_INT	; x0B
	.FILL BAD_INT	; x0C
	.FILL BAD_INT	; x0D
	.FILL BAD_INT	; x0E
	.FILL BAD_INT	; x0F
	.FILL BAD_INT	; x10
	.FILL BAD_INT	; x11
	.FILL BAD_INT	; x12
	.FILL BAD_INT	; x13
	.FILL BAD_INT	; x14
	.FILL BAD_INT	; x15
	.FILL BAD_INT	; x16
	.FILL BAD_INT	; x17
	.FILL BAD_INT	; x18
	.FILL BAD_INT	; x19
	.FILL BAD_INT	; x1A
	.FILL BAD_INT	; x1B
	.FILL BAD_INT	; x1C
	.FILL BAD_INT	; x1D
	.FILL BAD_INT	; x1E
	.FILL BAD_INT	; x1F
	.FILL BAD_INT	; x20
	.FILL BAD_INT	; x21
	.FILL BAD_INT	; x22
	.FILL BAD_INT	; x23
	.FILL BAD_INT   ; x24
	.FILL BAD_INT	; x25
	.FILL BAD_INT	; x26
	.FILL BAD_INT	; x27
	.FILL BAD_INT	; x28
	.FILL BAD_INT	; x29
	.FILL BAD_INT	; x2A
	.FILL BAD_INT	; x2B
	.FILL BAD_INT	; x2C
	.FILL BAD_INT	; x2D
	.FILL BAD_INT	; x2E
	.FILL BAD_INT	; x2F
	.FILL BAD_INT	; x30
	.FILL BAD_INT	; x31
	.FILL BAD_INT	; x32
	.FILL BAD_INT	; x33
	.FILL BAD_INT	; x34
	.FILL BAD_INT	; x35
	.FILL BAD_INT	; x36
	.FILL BAD_INT	; x37
	.FILL BAD_INT	; x38
	.FILL BAD_INT	; x39
	.FILL BAD_INT	; x3A
	.FILL BAD_INT	; x3B
	.FILL BAD_INT	; x3C
	.FILL BAD_INT	; x3D
	.FILL BAD_INT	; x3E
	.FILL BAD_INT	; x3F
	.FILL BAD_INT	; x40
	.FILL BAD_INT	; x41
	.FILL BAD_INT	; x42
	.FILL BAD_INT	; x43
	.FILL BAD_INT	; x44
	.FILL BAD_INT	; x45
	.FILL BAD_INT	; x46
	.FILL BAD_INT	; x47
	.FILL BAD_INT	; x48
	.FILL BAD_INT	; x49
	.FILL BAD_INT	; x4A
	.FILL BAD_INT	; x4B
	.FILL BAD_INT	; x4C
	.FILL BAD_INT	; x4D
	.FILL BAD_INT	; x4E
	.FILL BAD_INT	; x4F
	.FILL BAD_INT	; x50
	.FILL BAD_INT	; x51
	.FILL BAD_INT	; x52
	.FILL BAD_INT	; x53
	.FILL BAD_INT	; x54
	.FILL BAD_INT	; x55
	.FILL BAD_INT	; x56
	.FILL BAD_INT	; x57
	.FILL BAD_INT	; x58
	.FILL BAD_INT	; x59
	.FILL BAD_INT	; x5A
	.FILL BAD_INT	; x5B
	.FILL BAD_INT	; x5C
	.FILL BAD_INT	; x5D
	.FILL BAD_INT	; x5E
	.FILL BAD_INT	; x5F
	.FILL BAD_INT	; x60
	.FILL BAD_INT	; x61
	.FILL BAD_INT	; x62
	.FILL BAD_INT	; x63
	.FILL BAD_INT	; x64
	.FILL BAD_INT	; x65
	.FILL BAD_INT	; x66
	.FILL BAD_INT	; x67
	.FILL BAD_INT	; x68
	.FILL BAD_INT	; x69
	.FILL BAD_INT	; x6A
	.FILL BAD_INT	; x6B
	.FILL BAD_INT	; x6C
	.FILL BAD_INT	; x6D
	.FILL BAD_INT	; x6E
	.FILL BAD_INT	; x6F
	.FILL BAD_INT	; x70
	.FILL BAD_INT	; x71
	.FILL BAD_INT	; x72
	.FILL BAD_INT	; x73
	.FILL BAD_INT	; x74
	.FILL BAD_INT	; x75
	.FILL BAD_INT	; x76
	.FILL BAD_INT	; x77
	.FILL BAD_INT	; x78
	.FILL BAD_INT	; x79
	.FILL BAD_INT	; x7A
	.FILL BAD_INT	; x7B
	.FILL BAD_INT	; x7C
	.FILL BAD_INT	; x7D
	.FILL BAD_INT	; x7E
	.FILL BAD_INT	; x7F
	.FILL BAD_INT	; x80
	.FILL BAD_INT	; x81
	.FILL BAD_INT	; x82
	.FILL BAD_INT	; x83
	.FILL BAD_INT	; x84
	.FILL BAD_INT	; x85
	.FILL BAD_INT	; x86
	.FILL BAD_INT	; x87
	.FILL BAD_INT	; x88
	.FILL BAD_INT	; x89
	.FILL BAD_INT	; x8A
	.FILL BAD_INT	; x8B
	.FILL BAD_INT	; x8C
	.FILL BAD_INT	; x8D
	.FILL BAD_INT	; x8E
	.FILL BAD_INT	; x8F
	.FILL BAD_INT	; x90
	.FILL BAD_INT	; x91
	.FILL BAD_INT	; x92
	.FILL BAD_INT	; x93
	.FILL BAD_INT	; x94
	.FILL BAD_INT	; x95
	.FILL BAD_INT	; x96
	.FILL BAD_INT	; x97
	.FILL BAD_INT	; x98
	.FILL BAD_INT	; x99
	.FILL BAD_INT	; x9A
	.FILL BAD_INT	; x9B
	.FILL BAD_INT	; x9C
	.FILL BAD_INT	; x9D
	.FILL BAD_INT	; x9E
	.FILL BAD_INT	; x9F
	.FILL BAD_INT	; xA0
	.FILL BAD_INT	; xA1
	.FILL BAD_INT	; xA2
	.FILL BAD_INT	; xA3
	.FILL BAD_INT	; xA4
	.FILL BAD_INT	; xA5
	.FILL BAD_INT	; xA6
	.FILL BAD_INT	; xA7
	.FILL BAD_INT	; xA8
	.FILL BAD_INT	; xA9
	.FILL BAD_INT	; xAA
	.FILL BAD_INT	; xAB
	.FILL BAD_INT	; xAC
	.FILL BAD_INT	; xAD
	.FILL BAD_INT	; xAE
	.FILL BAD_INT	; xAF
	.FILL BAD_INT	; xB0
	.FILL BAD_INT	; xB1
	.FILL BAD_INT	; xB2
	.FILL BAD_INT	; xB3
	.FILL BAD_INT	; xB4
	.FILL BAD_INT	; xB5
	.FILL BAD_INT	; xB6
	.FILL BAD_INT	; xB7
	.FILL BAD_INT	; xB8
	.FILL BAD_INT	; xB9
	.FILL BAD_INT	; xBA
	.FILL BAD_INT	; xBB
	.FILL BAD_INT	; xBC
	.FILL BAD_INT	; xBD
	.FILL BAD_INT	; xBE
	.FILL BAD_INT	; xBF
	.FILL BAD_INT	; xC0
	.FILL BAD_INT	; xC1
	.FILL BAD_INT	; xC2
	.FILL BAD_INT	; xC3
	.FILL BAD_INT	; xC4
	.FILL BAD_INT	; xC5
	.FILL BAD_INT	; xC6
	.FILL BAD_INT	; xC7
	.FILL BAD_INT	; xC8
	.FILL BAD_INT	; xC9
	.FILL BAD_INT	; xCA
	.FILL BAD_INT	; xCB
	.FILL BAD_INT	; xCC
	.FILL BAD_INT	; xCD
	.FILL BAD_INT	; xCE
	.FILL BAD_INT	; xCF
	.FILL BAD_INT	; xD0
	.FILL BAD_INT	; xD1
	.FILL BAD_INT	; xD2
	.FILL BAD_INT	; xD3
	.FILL BAD_INT	; xD4
	.FILL BAD_INT	; xD5
	.FILL BAD_INT	; xD6
	.FILL BAD_INT	; xD7
	.FILL BAD_INT	; xD8
	.FILL BAD_INT	; xD9
	.FILL BAD_INT	; xDA
	.FILL BAD_INT	; xDB
	.FILL BAD_INT	; xDC
	.FILL BAD_INT	; xDD
	.FILL BAD_INT	; xDE
	.FILL BAD_INT	; xDF
	.FILL BAD_INT	; xE0
	.FILL BAD_INT	; xE1
	.FILL BAD_INT	; xE2
	.FILL BAD_INT	; xE3
	.FILL BAD_INT	; xE4
	.FILL BAD_INT	; xE5
	.FILL BAD_INT	; xE6
	.FILL BAD_INT	; xE7
	.FILL BAD_INT	; xE8
	.FILL BAD_INT	; xE9
	.FILL BAD_INT	; xEA
	.FILL BAD_INT	; xEB
	.FILL BAD_INT	; xEC
	.FILL BAD_INT	; xED
	.FILL BAD_INT	; xEE
	.FILL BAD_INT	; xEF
	.FILL BAD_INT	; xF0
	.FILL BAD_INT	; xF1
	.FILL BAD_INT	; xF2
	.FILL BAD_INT	; xF3
	.FILL BAD_INT	; xF4
	.FILL BAD_INT	; xF5
	.FILL BAD_INT	; xF6
	.FILL BAD_INT	; xF7
	.FILL BAD_INT	; xF8
	.FILL BAD_INT	; xF9
	.FILL BAD_INT	; xFA
	.FILL BAD_INT	; xFB
	.FILL BAD_INT	; xFC
	.FILL BAD_INT	; xFD
	.FILL BAD_INT	; xFE
	.FILL BAD_INT	; xFF


OS_START	; machine starts executing at x0200
	LEA R0,OS_START_MSG	; print a welcome message
	PUTS
	HALT

OS_START_MSG	.STRINGZ "\\nWelcome to the LC-3 simulator.\\n\\nThe contents of the LC-3 tools distribution, including sources, management\\ntools, and data, are Copyright (c) 2003 Steven S. Lumetta.\\n\\nThe LC-3 tools distribution is free software covered by the GNU General\\nPublic License, and you are welcome to modify it and/or distribute copies\\nof it under certain conditions.  The file COPYING (distributed with the\\ntools) specifies those conditions.  There is absolutely no warranty for\\nthe LC-3 tools distribution, as described in the file NO_WARRANTY (also\\ndistributed with the tools).\\n\\nHave fun.\\n"

OS_KBSR	.FILL xFE00
OS_KBDR	.FILL xFE02
OS_DSR	.FILL xFE04
OS_DDR	.FILL xFE06
OS_MCR	.FILL xFFFE
MASK_HI .FILL x7FFF
LOW_8_BITS .FILL x00FF
TOUT_R1 .BLKW 1
TIN_R7  .BLKW 1
OS_R0   .BLKW 1
OS_R1   .BLKW 1
OS_R2   .BLKW 1
OS_R3   .BLKW 1
OS_R7   .BLKW 1


TRAP_GETC
	LDI R0,OS_KBSR		; wait for a keystroke
	BRzp TRAP_GETC
	LDI R0,OS_KBDR		; read it and return
	RET

TRAP_OUT
	ST R1,TOUT_R1		; save R1
TRAP_OUT_WAIT
	LDI R1,OS_DSR		; wait for the display to be ready
	BRzp TRAP_OUT_WAIT
	STI R0,OS_DDR		; write the character and return
	LD R1,TOUT_R1		; restore R1
	RET

TRAP_PUTS
	ST R0,OS_R0		; save R0, R1, and R7
	ST R1,OS_R1
	ST R7,OS_R7
	ADD R1,R0,#0		; move string pointer (R0) into R1

TRAP_PUTS_LOOP
	LDR R0,R1,#0		; write characters in string using OUT
	BRz TRAP_PUTS_DONE
	OUT
	ADD R1,R1,#1
	BRnzp TRAP_PUTS_LOOP

TRAP_PUTS_DONE
	LD R0,OS_R0		; restore R0, R1, and R7
	LD R1,OS_R1
	LD R7,OS_R7
	RET

TRAP_IN
	ST R7,TIN_R7		; save R7 (no need to save R0, since we 
				;    overwrite later
	LEA R0,TRAP_IN_MSG	; prompt for input
	PUTS
	GETC			; read a character
	OUT			; echo back to monitor
	ST R0,OS_R0		; save the character
	AND R0,R0,#0		; write a linefeed, too
	ADD R0,R0,#10
	OUT
	LD R0,OS_R0		; restore the character
	LD R7,TIN_R7		; restore R7
	RET

TRAP_PUTSP
	; NOTE: This trap will end when it sees any NUL, even in
	; packed form, despite the P&P second edition's requirement
	; of a double NUL.

	ST R0,OS_R0		; save R0, R1, R2, R3, and R7
	ST R1,OS_R1
	ST R2,OS_R2
	ST R3,OS_R3
	ST R7,OS_R7
	ADD R1,R0,#0		; move string pointer (R0) into R1

TRAP_PUTSP_LOOP
	LDR R2,R1,#0		; read the next two characters
	LD R0,LOW_8_BITS	; use mask to get low byte
	AND R0,R0,R2		; if low byte is NUL, quit printing
	BRz TRAP_PUTSP_DONE
	OUT			; otherwise print the low byte

	AND R0,R0,#0		; shift high byte into R0
	ADD R3,R0,#8
TRAP_PUTSP_S_LOOP
	ADD R0,R0,R0		; shift R0 left
	ADD R2,R2,#0		; move MSB from R2 into R0
	BRzp TRAP_PUTSP_MSB_0
	ADD R0,R0,#1
TRAP_PUTSP_MSB_0
	ADD R2,R2,R2		; shift R2 left
	ADD R3,R3,#-1
	BRp TRAP_PUTSP_S_LOOP

	ADD R0,R0,#0		; if high byte is NUL, quit printing
	BRz TRAP_PUTSP_DONE
	OUT			; otherwise print the low byte

	ADD R1,R1,#1		; and keep going
	BRnzp TRAP_PUTSP_LOOP

TRAP_PUTSP_DONE
	LD R0,OS_R0		; restore R0, R1, R2, R3, and R7
	LD R1,OS_R1
	LD R2,OS_R2
	LD R3,OS_R3
	LD R7,OS_R7
	RET

TRAP_HALT	
	; an infinite loop of lowering OS_MCR's MSB
	LEA R0,TRAP_HALT_MSG	; give a warning
	PUTS
	LDI R0,OS_MCR		; halt the machine
	LD R1,MASK_HI
	AND R0,R0,R1
	STI R0,OS_MCR
	BRnzp TRAP_HALT		; HALT again...

BAD_TRAP
	; print an error message, then HALT
	LEA R0,BAD_TRAP_MSG	; give an error message
	PUTS
	BRnzp TRAP_HALT		; execute HALT

	; interrupts aren't really defined, since privilege doesn't
	; quite work
INT_PRIV	RTI
INT_ILL		RTI
BAD_INT		RTI

TRAP_IN_MSG	.STRINGZ "\\nInput a character> "
TRAP_HALT_MSG	.STRINGZ "\\n\\n--- halting the LC-3 ---\\n\\n"
BAD_TRAP_MSG	.STRINGZ "\\n\\n--- undefined trap executed ---\\n\\n"

	.END


EOF
cat>lc3sim-tk.def<<EOF
#!/bin/sh
# the next line restarts using wish \\
exec @@WISH@@ "\$0" -- "\$@"

###############################################################################
#
# lc3sim-tk -- a Tcl/Tk graphical front end to the LC-3 simulator
#
#  "Copyright (c) 2003 by Steven S. Lumetta."
# 
#  Permission to use, copy, modify, and distribute this software and its
#  documentation for any purpose, without fee, and without written 
#  agreement is hereby granted, provided that the above copyright notice
#  and the following two paragraphs appear in all copies of this software,
#  that the files COPYING and NO_WARRANTY are included verbatim with
#  any distribution, and that the contents of the file README are included
#  verbatim as part of a file named README with any distribution.
#  
#  IN NO EVENT SHALL THE AUTHOR BE LIABLE TO ANY PARTY FOR DIRECT, 
#  INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT 
#  OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE AUTHOR 
#  HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#  
#  THE AUTHOR SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT 
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR 
#  A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED HEREUNDER IS ON AN "AS IS" 
#  BASIS, AND THE AUTHOR NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, 
#  UPDATES, ENHANCEMENTS, OR MODIFICATIONS."
#
#  Author:		Steve Lumetta
#  Version:		1
#  Creation Date:	18 October 2003
#  Filename:		lc3sim-tk
#  History:		
# 	SSL	1	18 October 2003
# 		Copyright notices and Gnu Public License marker added.
#
###############################################################################

set path(lc3sim) @@LC3_SIM@@

# If an error occurs, exit without showing the window.
wm withdraw .
proc set_default_options {} {
    global option

    set option(code_width)     60
    set option(code_height)    40
    set option(code_font)      @@CODE_FONT@@
    set option(code_bg)        Blue4
    set option(code_fg)        Yellow
    set option(pc_bg)          \$option(code_fg)
    set option(pc_fg)          \$option(code_bg)
    set option(break_bg)       Red
    set option(break_fg)       White
    set option(button_font)    @@BUTTON_FONT@@

    set option(console_width)  80
    set option(console_height) 25
    set option(console_font)   @@CONSOLE_FONT@@
    set option(console_bg)     \$option(code_bg)
    set option(console_fg)     \$option(code_fg)

    set option(flush)          on
    set option(device)         on
    set option(delay)          on
}


# Fills variables and widgets in options window with values from 
# option strings.  Used when options window is created and when values
# are restored to default values (which are only known as strings).

proc fill_option_window {} {
    global opt_list option optval

    foreach opt \$opt_list {
	set name  [lindex \$opt 0]
        set optval(\$name) \$option(\$name)
	switch -exact [lindex \$opt 2] {
	    font {
		set optval(\$name,f) [lindex \$option(\$name) 0]
		set optval(\$name,p) [lindex \$option(\$name) 1]
		set optval(\$name,b) [lindex \$option(\$name) 2]
	    }
	}
    }
}

# Using data from the corresponding widget(s) in the options window,
# apply a single option and create an appropriate option string for
# that option.  Used when individual options are set and indirectly
# whenever all options are saved or applied.

proc apply_option {opt} {
    global option optval

    set name  [lindex \$opt 0]
    set plist [lindex \$opt 3]
    switch -exact [lindex \$opt 2] {
	font {
	    set optval(\$name) \\
	    	[list \$optval(\$name,f) \$optval(\$name,p) \$optval(\$name,b)]
	}
	color {
	    catch {.options.\$name.b config -bg \$optval(\$name)}
	}
    }
    foreach item \$plist {
	eval "[lindex \$item 0] config [lindex \$item 1] \\"\$optval(\$name)\\""
	set option(\$name) \$optval(\$name)
    }
}

# Applies all options as above, then writes simulator options to the
# simulator.  Used when "Apply" or "Save" [options] are pressed, and
# when default settings are restored.

proc apply_all_options {} {
    global opt_list

    foreach opt \$opt_list {apply_option \$opt}
    apply_sim_options
}

proc apply_sim_options {} {
    global sim option

    # Note that the "keep" option in lc3sim is only meaningful for 
    # terminal input; since the GUI separates the input channels, there 
    # is no chance of LC-3 input being used as simulator commands, and 
    # thus no point in not keeping it.  Turning keep off and flush are 
    # also equivalent in the GUI, so the functionality is still available.

    foreach name {flush device delay} {
	puts \$sim "option \$name \$option(\$name)"
    }
}

proc write_option_file {} {
    global opt_list option

    if {[catch {
	set ofile [open "~/.lc3simrc" w]
	apply_all_options
	foreach opt \$opt_list {
	    set name [lindex \$opt 0]
	    puts \$ofile "set option(\$name) \\"\$option(\$name)\\""
	}
	foreach name {flush device delay} {
	    puts \$ofile "set option(\$name) \\"\$option(\$name)\\""
	}
	close \$ofile
    } msg]} {
        tk_messageBox -message \$msg
    } {
        tk_messageBox -message "Options saved." -parent .options
    }
}

proc fontmenu {oname mname opt} {
    menu \$mname -tearoff 0
    foreach font [font families] {
	if {[font metrics [list \$font] -fixed]} {
	    \$mname add command -label "\$font" -font [list \$font] -command "
		set optval(\$oname,f) \\"\$font\\"
		apply_option [list \$opt]
	    "
	}
    }
    \$mname add separator
    foreach font [font families] {
	if {![font metrics [list \$font] -fixed]} {
	    \$mname add command -label "\$font" -font [list \$font] -command "
		set optval(\$oname,f) \\"\$font\\"
		apply_option [list \$opt]
	    "
	}
    }
}

proc stylemenu {oname mname opt} {
    menu \$mname -tearoff 0
    foreach style {normal bold italic} {
	\$mname add command -label \$style -command "
	    set optval(\$oname,b) \$style
	    apply_option [list \$opt]
	"
    }
}

proc change_options {} {
    global opt_list option optval

    if {![winfo exists .options]} {
	show_delay 1
	update
        toplevel .options
	wm title .options "LC-3 Simulator Options"
	set opt_list {
	    {code_width     "Code Width"      int   {{.code -width}}}
	    {code_height    "Code Height"     int   {{.code -height}}}
	    {code_font      "Code Font"       font
	     {{.code -font} {.regs.0.0.l -font} {.regs.0.0.e -font}
	      {.regs.0.1.l -font} {.regs.0.1.e -font}
	      {.regs.0.2.l -font} {.regs.0.2.e -font}
	      {.regs.1.0.l -font} {.regs.1.0.e -font}
	      {.regs.1.1.l -font} {.regs.1.1.e -font}
	      {.regs.1.2.l -font} {.regs.1.2.e -font}
	      {.regs.2.0.l -font} {.regs.2.0.e -font}
	      {.regs.2.1.l -font} {.regs.2.1.e -font}
	      {.regs.2.2.l -font} {.regs.2.2.e -font}
	      {.regs.3.0.l -font} {.regs.3.0.e -font}
	      {.regs.3.1.l -font} {.regs.3.1.e -font}
	      {.regs.3.2.l -font} {.regs.3.2.e -font}
	      {.mem.l -font} {.mem.addr -font}
	      {.mem.value -font} {.mem.l2 -font}
	      {.in.l -font} {.in.e -font}
	     }}
	    {code_bg        "Code Background" color {{.code -bg}}}
	    {code_fg        "Code Foreground" color {{.code -fg}}}
	    {pc_bg          "PC Background"   color   
	     {{{.code tag} {pc_at -background}}}}
	    {pc_fg          "PC Foreground"   color
	     {{{.code tag} {pc_at -foreground}}}}
	    {break_bg       "Breakpoint Background" color
	     {{{.code tag} {break -background}}}}
	    {break_fg       "Breakpoint Foreground" color
	     {{{.code tag} {break -foreground}}}}
	    {button_font    "Button Font"     font  
	     {{.ctrl.n -font} {.ctrl.s -font} {.ctrl.x -font} {.ctrl.c -font}
	      {.ctrl.f -font} {.ctrl.p -font} {.ctrl.bca -font} 
	      {.in.browse -font} 
	      {.misc.reset -font} {.misc.options -font} {.misc.quit -font}
	     }}
	    {console_width  "Console Width"   int   {{.console.t -width}}}
	    {console_height "Console Height"  int   {{.console.t -height}}}
	    {console_font   "Console Font"    font  {{.console.t -font}}}
	    {console_bg     "Console Background" color
	     {{.console.t -bg}}}
	    {console_fg     "Console Foreground" color
	     {{.console.t -fg}}}
	}

	frame .options.ctrltop
	button .options.ctrltop.def -width 19 -text "Use Default Values" \\
	    -command {
	    set_default_options
	    fill_option_window
	    apply_all_options
	}
	pack .options.ctrltop.def -side right -padx 5
	pack .options.ctrltop -side top -expand t -fill x -pady 5
	fill_option_window
	foreach opt \$opt_list {
	    set name    [lindex \$opt 0]
	    set visname [lindex \$opt 1]
	    set type    [lindex \$opt 2]
	    set plist   [lindex \$opt 3]
	    frame .options.\$name
	    label .options.\$name.l -width 25 -anchor e -text \$visname 
	    pack .options.\$name.l -side left
	    set optval(\$name) \$option(\$name)
	    switch -exact \$type {
	    	int {
		    entry .options.\$name.e -width 20 \\
		    	-textvariable optval(\$name)
		    bind .options.\$name.e <KeyPress-Return> "
		    	apply_option [list \$opt]
		    "
		    pack .options.\$name.e -side left -expand t -fill x -padx 5
		}
		font {
		    menubutton .options.\$name.m -width 15 \\
		    	-menu .options.\$name.m.m -textvariable optval(\$name,f)
		    fontmenu \$name .options.\$name.m.m \$opt
		    entry .options.\$name.p -width 2 \\
		    	-textvariable optval(\$name,p)
		    set optval(\$name,p) [lindex \$optval(\$name) 1]
		    bind .options.\$name.p <KeyPress-Return> "
		        apply_option [list \$opt]
		    "
		    menubutton .options.\$name.b -width 8 \\
		    	-menu .options.\$name.b.m -textvariable optval(\$name,b)
		    stylemenu \$name .options.\$name.b.m \$opt
		    pack .options.\$name.m -side left -padx 5
		    pack .options.\$name.b -side right -padx 5 
		    pack .options.\$name.p -expand t -fill x
		}
		color {
		    button .options.\$name.b -width 20 -bg \$optval(\$name) \\
		    	-command "
		        set acolor \\
			    \\[tk_chooseColor -initialcolor \\\$optval(\$name) \\
			      -parent .options\\]
			if {\\\$acolor != {}} {
			    set optval(\$name) \\\$acolor
			    apply_option [list \$opt]
			    catch {.options.\$name.b config -bg \\\$optval(\$name)}
			}
		    " 
		    pack .options.\$name.b -side left -expand t -fill x -padx 5
		}
	    }
	    pack .options.\$name -side top -expand t -fill x -pady 2
	}

	checkbutton .options.flush -variable option(flush) \\
	    -onvalue on -offvalue off \\
	    -text "Flush LC-3 console input when restarting"
	pack .options.flush -side top -anchor w -pady 2 

	checkbutton .options.device -variable option(device) \\
	    -onvalue on -offvalue off \\
	    -text "Simulate random timing for device registers"
	pack .options.device -side top -anchor w -pady 2 

	checkbutton .options.delay -variable option(delay) \\
	    -onvalue on -offvalue off \\
	    -text "Delay display of memory updates until LC-3 stops"
	pack .options.delay -side top -anchor w -pady 2 

	frame .options.ctrl
	button .options.ctrl.save -width 5 -text Save -command {
	    apply_all_options
	    write_option_file
	}
	button .options.ctrl.apply -width 6 -text Apply \\
		-command apply_all_options
	button .options.ctrl.close -width 6 -text Close \\
		-command {wm withdraw .options}
	pack .options.ctrl.save -side left -padx 5
	pack .options.ctrl.apply -side left -padx 5
	pack .options.ctrl.close -side right -padx 5
	pack .options.ctrl -side bottom -expand t -fill x -pady 5
	show_delay 0
    } {
        wm deiconify .options
    }
}

# set options
set_default_options
if {[file exists ~/.lc3simrc]} {
    source ~/.lc3simrc
}

# set up names for register buttons and entries
for {set i 0} {\$i < 12} {incr i} {
    set reg(name,\$i) R\$i
    set reg(visname,\$i) R\$i
}
set reg(visname,8)  PC
set reg(visname,9)  IR
set reg(visname,10) PSR
set reg(visname,11) CC

# create array of buttons and entries...
frame .regs
for {set i 0} {\$i < 4} {incr i} {
    frame .regs.\$i
    for {set j 0} {\$j < 3} {incr j} {
	set f .regs.\$i.\$j
	set n [expr {\$i + \$j * 4}]
	frame \$f
	label \$f.l -text \$reg(visname,\$n) -font \$option(code_font) -width 3
	entry \$f.e -width 10 -font \$option(code_font) \\
		-textvariable reg(\$reg(name,\$n))
	bind \$f.e <KeyPress-Return> "
	    puts \\\$sim \\"r \$reg(visname,\$n) \\\$reg(\$reg(name,\$n))\\"
	"
	pack \$f.l -side left
	pack \$f.e
	pack \$f -side top -pady 5
    }
    pack .regs.\$i -side left -expand t -padx 10
}
pack .regs -side top -fill x -pady 5

text .code -width \$option(code_width) -height \$option(code_height) \\
	-font \$option(code_font) -takefocus 1 -cursor arrow \\
	-bg \$option(code_bg) -fg \$option(code_fg) -wrap none \\
	-yscrollcommand {.code_y set} -state disabled

.code tag configure break -background \$option(break_bg) \\
	-foreground \$option(break_fg)
.code tag configure pc_at -background \$option(pc_bg) -foreground \$option(pc_fg)
scrollbar .code_y -command {.code yview} -orient vertical


# ctrl is translate, jump, or set
proc issue_trans_cmd {ctrl} {
    global sim mem

    set mem(ctrl) \$ctrl
    puts \$sim "t \$mem(addr)"
}

proc set_mem {} {
    global sim mem

    issue_trans_cmd set
}

frame .mem
label .mem.l -width 15 -text "Memory Address" -font \$option(code_font)
entry .mem.addr -width 18 -font \$option(code_font) -textvariable mem(addr)
bind .mem.addr <KeyPress-Return> {issue_trans_cmd jump}
frame .mem.pad -width 20
entry .mem.value -width 18 -font \$option(code_font) -textvariable mem(value)
label .mem.l2 -width 6 -text "Value" -font \$option(code_font)
bind .mem.value <KeyPress-Return> set_mem
pack .mem.l -side left
pack .mem.addr -side left
pack .mem.pad -side left
pack .mem.l2 -side left
pack .mem.value -side left
pack .mem -side top -fill x -pady 5

frame .ctrl
button .ctrl.n -text Next -command {puts \$sim n} -font \$option(button_font)
button .ctrl.s -text Step -command {puts \$sim s} -font \$option(button_font)
button .ctrl.f -text Finish -command {puts \$sim fin} -font \$option(button_font)
button .ctrl.c -text Continue -command {puts \$sim c} -font \$option(button_font)
button .ctrl.x -text Stop -command {puts \$sim x} -font \$option(button_font) \\
	-state disabled
button .ctrl.p -text "Update Registers" -command {puts \$sim p} \\
	-font \$option(button_font)
button .ctrl.bca -text "Clear All Breakpoints" -command clear_all_breakpoints \\
	-font \$option(button_font) -state disabled
pack .ctrl.n -side left -padx 5
pack .ctrl.s -side left -padx 5
pack .ctrl.f -side left -padx 5
pack .ctrl.c -side left -padx 5
pack .ctrl.x -side left -padx 5
pack .ctrl.p -side right -padx 5
pack .ctrl.bca -side right -padx 5
pack .ctrl -side top -fill x -pady 5

frame .rctrl
button .rctrl.rn -text rNext -command {puts \$sim rn} -font \$option(button_font)
button .rctrl.rs -text rStep -command {puts \$sim rs} -font \$option(button_font)
button .rctrl.rf -text rFinish -command {puts \$sim rfin} -font \$option(button_font)
button .rctrl.rc -text rContinue -command {puts \$sim rc} -font \$option(button_font)
pack .rctrl.rn -side left -padx 5
pack .rctrl.rs -side left -padx 5
pack .rctrl.rf -side left -padx 5
pack .rctrl.rc -side left -padx 5
pack .rctrl -side top -fill x -pady 5

frame .misc
button .misc.reset -text "Reset LC-3" -font \$option(button_font) \\
    -command reset_machine
button .misc.options -text Options -font \$option(button_font) \\
    -command change_options
button .misc.quit -text Quit -font \$option(button_font) -command {set halt 1}
pack .misc.reset -side left -padx 5
pack .misc.quit -side right -padx 5
pack .misc.options -padx 5
pack .misc -side bottom -fill x -pady 5

frame .in
label .in.l -width 13 -text "File to Load" -font \$option(code_font)
entry .in.e -width 25 -font \$option(code_font) -textvariable file
bind .in.e <KeyPress-Return> load_file
button .in.browse -text Browse -font \$option(button_font) -command pick_file
pack .in.l -side left
pack .in.browse -side right -padx 5
pack .in.e -expand t -fill x -padx 5
pack .in -side bottom -fill x -pady 5

pack .code_y -side right -fill y
pack .code -expand t -fill both

set save_curs(saved) 0
proc show_delay {yes} {
    global save_curs

    if {\$yes} {
	if {!\$save_curs(saved)} {
	    set save_curs(file)    [lindex [.in.e      config -cursor] 4]
	    set save_curs(code)    [lindex [.code      config -cursor] 4]
	    set save_curs(sim)     [lindex [.          config -cursor] 4]
	    set save_curs(console) [lindex [.console   config -cursor] 4]
	    set save_curs(con_txt) [lindex [.console.t config -cursor] 4]
	    set save_curs(saved) 1
	}
	.in.e      config -cursor watch
	.code      config -cursor watch
	.          config -cursor watch
	.console   config -cursor watch
	.console.t config -cursor watch
	return
    }
    if {\$save_curs(saved)} {
	.in.e      config -cursor \$save_curs(file)
	.code      config -cursor \$save_curs(code)
	.          config -cursor \$save_curs(sim)
	.console   config -cursor \$save_curs(console)
	.console.t config -cursor \$save_curs(con_txt)
	set save_curs(saved) 0
    }
}

proc load_file {} {
    global sim file

    show_delay 1
    puts \$sim "f \$file"
}

proc pick_file {} {
    global file

    set f [tk_getOpenFile -defaultextension .obj \\
	-filetypes {{"Object File" .obj}} -initialdir . \\
	-initialfile \$file -title "Load which file?"]
    if {\$f != ""} {
	set file \$f
	load_file
    }
}

set bpoints {}

proc set_mem_addr {index} {
    global mem

    scan \$index "%d" line
    set mem(addr) [format "x%x" [expr {\$line - 1}]]
    issue_trans_cmd translate
}

proc break_code_line {index} {
    global sim bpoints

    scan \$index "%d" line
    set addr [format "%x" [expr {\$line - 1}]]
    if {[lsearch -exact \$bpoints \${line}.0] == -1} {
        puts \$sim "b s x\$addr"
    } {
        puts \$sim "b c x\$addr"
    }
}

# kind of lazy here...could pull value from line in .code ... --SSL
bind .code <1> {
    set_mem_addr [.code index @%x,%y]
    focus .code
}
bind .code <Double-1> {break_code_line [.code index @%x,%y]}

# should never be used
set lc3_running 0  
proc highlight_pc {running} {
    global reg option lc3_running

    # record state of last call to allow re-highlighting
    set lc3_running \$running

    .code tag remove pc_at 1.0 end
    if {\$running} {
        .ctrl.n configure -state disabled
        .ctrl.s configure -state disabled
        .ctrl.f configure -state disabled
        .ctrl.c configure -state disabled
        .ctrl.x configure -state normal
	bind .code <KeyPress-n> {}
	bind .code <KeyPress-s> {}
	bind .code <KeyPress-f> {}
	bind .code <KeyPress-c> {}
    } {
	scan \$reg(R8) "x%x" pc
	set pcline [expr {\$pc + 1}].0
	.code tag add pc_at \$pcline "\$pcline +1 line"
	.code see \$pcline
        .ctrl.n configure -state normal
        .ctrl.s configure -state normal
        .ctrl.f configure -state normal
        .ctrl.c configure -state normal
        .ctrl.x configure -state disabled
	bind .code <KeyPress-n> {puts \$sim next}
	bind .code <KeyPress-s> {puts \$sim step}
	bind .code <KeyPress-f> {puts \$sim finish}
	bind .code <KeyPress-c> {puts \$sim continue}
    }
}

proc lc3sim_died {} {
    global sim lc3_console halt

    fileevent \$sim readable {}
    fileevent \$lc3_console readable {}
    catch {close \$sim}
    catch {close \$lc3_console}
    set halt 1
    error "The LC-3 simulator died."
}

proc clear_break {lnum} {
    global bpoints

    set pos [lsearch -exact \$bpoints \$lnum]
    set bpoints [lreplace \$bpoints \$pos \$pos]
    if {\$bpoints == ""} {
        .ctrl.bca configure -state disabled
    }
    .code configure -state normal
    .code tag remove break \$lnum "\$lnum +1 line"
    .code insert "\$lnum +1 char" " "
    .code delete \$lnum "\$lnum +1 char"
    .code configure -state disabled
}

proc clear_all_breakpoints {} {
    global sim bpoints

    puts \$sim "b c all"
    while {\$bpoints != ""} {clear_break [lindex \$bpoints 0]}
}

proc reset_machine {} {
    global sim

    # Easier to handle this way, although creates more work in simulator. 
    clear_all_breakpoints

    puts \$sim reset
    show_delay 1
}

proc insert_to_console {data} {
    .console.t configure -state normal
    .console.t insert end "\$data"
    .console.t configure -state disabled
    .console.t see end
}

proc delayed_read_lc3 {} {
    global lc3_console

    set line [read \$lc3_console]
    if {\$line == ""} {
    	return
    }

    insert_to_console "\$line"
}

proc read_lc3 {} {
    global lc3_console

    if {[gets \$lc3_console line] == -1} {
	if {[fblocked \$lc3_console]} {
	    # wait 1/4 second and read without waiting for newline
	    after 250 delayed_read_lc3
	    return
	}
	lc3sim_died
    }

    insert_to_console "\${line}\\n"
}

proc read_sim {} {
    global sim reg option bpoints mem fail_focus lc3_running

    if {[gets \$sim line] == -1} {
	if {[fblocked \$sim]} {return}
	lc3sim_died
    }

    # code disassembly may contain special characters, 
    # so don't treat it as a list
    if {[string range \$line 0 3] == "CODE"} {
	# format: 0-3 = CODE, 4 = P if PC, SPACE otherwise
	# 5-9 = line number in decimal, 10+ = code including B for breakpoint
        scan [string range \$line 5 9] %d lnum
        set lnum \${lnum}.0
	.code configure -state normal
	.code delete \$lnum "\$lnum +1 line"
	.code insert \$lnum [string range \$line 10 end]\\n
	.code configure -state disabled
	# pushed recognition duty onto the simulator...
	if {[string range \$line 4 4] == "P"} {
	    if {!\$lc3_running} {
		.code tag add pc_at \$lnum "\$lnum +1 line"
	    }
	}
	if {[string range \$line 10 10] == "B"} {
	    .code tag add break \$lnum "\$lnum +1 line"
	}
	return
    }

    set cmd [lindex \$line 0]

    if {\$cmd == "REG"} {
	set rnum [lindex \$line 1]
        set reg(\$rnum) [lindex \$line 2]
	if {\$rnum == "R8"} {highlight_pc 0}
	return
    }

    if {\$cmd == "CONT"} {
    	highlight_pc 1
	return
    }

    if {\$cmd == "BREAK"} {
        set lnum [lindex \$line 1].0
	lappend bpoints \$lnum
        .ctrl.bca configure -state normal
	.code configure -state normal
	.code insert "\$lnum +1 char" B
	.code delete \$lnum "\$lnum +1 char"
	.code tag add break \$lnum "\$lnum +1 line"
	.code configure -state disabled
	return
    }

    if {\$cmd == "BCLEAR"} {
        set lnum [lindex \$line 1].0
	clear_break \$lnum
	return
    }

    if {\$cmd == "TRANS"} {
	set arg1 [lindex \$line 1]
	set mem(addr) \$arg1
	if {\$mem(ctrl) == "jump"} {
	    scan \$arg1 "x%x" lnum
	    .code see [expr {\$lnum + 1}].0
	}
	if {\$mem(ctrl) == "set"} {
	    # store result of data translation into entry box
	    # memory set command also returns translation 
	    # to allow us to replace data labels with real values in the
	    # entry box
	    set mem(ctrl) translate
	    puts \$sim "m \$mem(addr) \$mem(value)"
	} {
	    set mem(value) [lindex \$line 2]
	    focus .code
	}
	return
    }

    # successful command--refocus to code window
    if {\$cmd == "TOCODE"} {
	show_delay 0
        focus .code
	return
    }

    if {\$cmd == "ERR"} {
	show_delay 0
        tk_messageBox -message [lindex \$line 1]
	return
    }

    # FIXME -- get rid of this debug thing
    #insert_to_console ":::\${line}:::\\n"
}

proc open_lc3channel {sock ip_addr port} {
    global lc3_listen lc3_console

    close \$lc3_listen
    unset lc3_listen
    if {\$ip_addr != "127.0.0.1"} {
        error "contacted by bad IP address (not loopback)"
    }
    set lc3_console \$sock
    fconfigure \$lc3_console -blocking 0 -buffering none
    fileevent \$lc3_console readable read_lc3
}

toplevel .console
wm withdraw .console
wm title .console "LC-3 Console"
wm title . "LC-3 Simulator Interface"

text .console.t -width \$option(console_width) -height \$option(console_height) \\
	-font \$option(console_font) -state disabled -takefocus 0 \\
	-bg \$option(console_bg) -fg \$option(console_fg) -wrap char \\
	-yscrollcommand {.console.t_y set} -cursor arrow
scrollbar .console.t_y -command {.console.t yview} -orient vertical
pack .console.t_y -side right -fill y
pack .console.t -expand t -fill both

bind .console <KeyPress> {
    if {"%A" != ""} {
	puts -nonewline \$lc3_console %A
    }
}

# could be annoying...
#bind . <KeyPress-Escape> {set halt 1}

bind . <Destroy> {exit}
bind .console <Destroy> {set halt 1}

set sim [open "| \$path(lc3sim) -gui \$argv" r+]
fconfigure \$sim -blocking 0 -buffering none

while {![info exists lc3_listen]} {
    set port [expr {int (rand () * 1000 + 5000)}]
    catch {set lc3_listen [socket -server open_lc3channel \$port]}
}

puts \$sim \$port

vwait lc3_listen
if {![info exists lc3_console]} {
    # show error message before exiting
    update  
    # avoid self-deadlock issues with some versions of WISH
    bind . <Destroy> {}
    bind .console <Destroy> {}
    exit
}

# pass options to simulator
if {[file exists ~/.lc3simrc]} {
    apply_sim_options
}

# fill memory with 0's, displaying after the text box first fills
.code configure -state normal
for {set i 0} {\$i < 236} {incr i} {
    .code insert end [format "                   x%04X x0000 NOP\\n" \$i]
}
wm deiconify .
wm deiconify .console
show_delay 1
update
for {} {\$i < 65536} {incr i} {
    .code insert end [format "                   x%04X x0000 NOP\\n" \$i]
}
.code delete 65537.0 65538.0
.code configure -state disabled
update

# now we're ready to pay attention to the simulator...
fileevent \$sim readable read_sim

focus .code
show_delay 0

catch {vwait halt}
update
fileevent \$sim readable {}
puts \$sim quit
# avoid self-deadlock issues with some versions of WISH
bind . <Destroy> {}
bind .console <Destroy> {}
exit

EOF
cat>lc3sim.c<<EOF
/*									tab:8
 *
 * lc3sim.c - the main source file for the LC-3 simulator
 *
 * "Copyright (c) 2003-2020 by Steven S. Lumetta and LIU Tingkai."
 *
 * Permission to use, copy, modify, and distribute this software and its
 * documentation for any purpose, without fee, and without written 
 * agreement is hereby granted, provided that the above copyright notice
 * and the following two paragraphs appear in all copies of this software,
 * that the files COPYING and NO_WARRANTY are included verbatim with
 * any distribution, and that the contents of the file README are included
 * verbatim as part of a file named README with any distribution.
 * 
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE TO ANY PARTY FOR DIRECT, 
 * INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT 
 * OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE AUTHOR 
 * HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 * THE AUTHOR SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT 
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR 
 * A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED HEREUNDER IS ON AN "AS IS" 
 * BASIS, AND THE AUTHOR NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, 
 * UPDATES, ENHANCEMENTS, OR MODIFICATIONS."
 *
 * Author:	    Steve Lumetta
 * Version:	    2
 * Creation Date:   18 October 2003
 * Filename:	    lc3sim.c
 * History:
 *	SSL	1	18 October 2003
 *		Copyright notices and Gnu Public License marker added.
 *	TKL/SSL	2	9 October 2020
 *		Integrated Tingkai Liu's extensions into main code.  Also
 *		changed strings to char* to avoid modern C warnings (safe
 *		enough with char warning for array index; mod'd isspace use).
 */

#include <ctype.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <strings.h>
#include <sys/poll.h>
#include <sys/termios.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <time.h>
#include <unistd.h>

#if defined(USE_READLINE)
#include <readline/readline.h>
#include <readline/history.h>
#endif

#ifdef WASM
#include <emscripten.h>
#endif

#include "lc3sim.h"
#include "symbol.h"

/* Disassembly format specification. */
#define OPCODE_WIDTH 6 

/* NOTE: hardcoded in scanfs! */
#define MAX_CMD_WORD_LEN    41    /* command word limit + 1 */
#define MAX_FILE_NAME_LEN  251    /* file name limit + 1    */
#define MAX_LABEL_LEN       81    /* label limit + 1        */

#define MAX_SCRIPT_DEPTH    10    /* prevent infinite recursion in scripts */
#define MAX_FINISH_DEPTH 10000000 /* avoid waiting to finish subroutine    */
				  /* that recurses infinitely              */

// Max number of cached steps for a program
#define MAX_CACHED_STEPS	0x20000 
// Maximum registers that can be changed by a instruction (PC, IR, PSR, DR)
#define MAX_REG_CHANGED 	4 	
// Maximum memory localtions that can be changed by a instruction
#define MAX_MEM_CHANGED		1 	

#define TOO_MANY_ARGS     "WARNING: Ignoring excess arguments."
#define BAD_ADDRESS       \\
	"Addresses must be labels or values in the range x0000 to xFFFF."

/* Types of cached events. */
typedef enum cache_type_t cache_type_t;
enum cache_type_t {
    CACHE_TYPE_NONE = -1, CACHE_TYPE_INST, CACHE_TYPE_MEM, CACHE_TYPE_REG, CACHE_TYPE_PC
};

/* 
   Types of breakpoints.  Currently only user breakpoints are
   handled in this manner; the system breakpoint used for the
   "next" command is specified by sys_bpt_addr.
*/
typedef enum bpt_type_t bpt_type_t;
enum bpt_type_t {BPT_NONE, BPT_USER};

typedef enum watch_type_t watch_type_t;
enum watch_type_t {WATCH_NONE, WATCH_USER};

static int launch_gui_connection ();
static char* simple_readline (const char* prompt);

static void init_machine ();
static void print_register (int which);
static void print_registers ();
static void dump_delayed_mem_updates ();
static void show_state_if_stop_visible ();
#ifdef WASM
static int read_obj_file(const unsigned char *filename, const unsigned char *debugname, int *startp, int *endp);
#else
static int read_obj_file (const char* filename, int* startp, int* endp);
#endif
static int read_sym_file (const char* filename);
static void squash_symbols (int addr_s, int addr_e);
static int execute_instruction ();
#ifdef WASM
static void disassemble_one (int addr, int should_report);
#else
static void disassemble_one (int addr);
#endif
static void disassemble (int addr_s, int addr_e);
static void dump_memory (int addr_s, int addr_e);
static void run_until_stopped ();
static void clear_breakpoint (int addr);
static void clear_all_breakpoints ();
static void list_breakpoints ();
static void set_breakpoint (int addr, int skip_count);
static void warn_too_many_args ();
static void no_args_allowed (const char* args);
static int parse_address (const char* addr);
static int parse_range (const char* cmd, int* startptr, int* endptr, 
			int last_end, int scale);
static void flush_console_input ();
static void gui_stop_and_dump ();


// Commands
static void cmd_break     (const char* args);
static void cmd_continue  (const char* args);
static void cmd_dump      (const char* args);
static void cmd_execute   (const char* args);
static void cmd_file      (const char* args);
static void cmd_finish    (const char* args);
static void cmd_help      (const char* args);
static void cmd_list      (const char* args);
static void cmd_memory    (const char* args);
static void cmd_next      (const char* args);
static void cmd_option    (const char* args);
static void cmd_printregs (const char* args);
static void cmd_quit      (const char* args);
static void cmd_register  (const char* args);
static void cmd_reset     (const char* args);
static void cmd_r_step	  (const char* args);
static void cmd_r_next	  (const char* args);
static void cmd_r_continue(const char* args);
static void cmd_r_finish  (const char* args);
static void cmd_step      (const char* args);
static void cmd_translate (const char* args);
static void cmd_lc3_stop  (const char* args);
// left out for now
//static void cmd_watch	  (const char* args);


// State cache management 
static void cache_ (cache_type_t type); 
static void clear_cache ();
static int cache_step (int step, cache_type_t type);
static int restore_step ();
static inst_flag_t cache_last_flags ();
static cache_type_t cache_last_type ();


/******************************** Data types *******************************/

typedef enum cmd_flag_t cmd_flag_t;
enum cmd_flag_t {
    CMD_FLAG_NONE       = 0,
    CMD_FLAG_REPEATABLE = 1, /* pressing ENTER repeats command  */
    CMD_FLAG_LIST_TYPE  = 2, /* pressing ENTER shows more       */
    CMD_FLAG_GUI_ONLY   = 4, /* only valid in GUI mode          */
    CMD_FLAG_REVERSE 	= 8  /* command for reverse execution   */
};

typedef struct command_t command_t;
struct command_t {
    char* command;  /* string for command                             */
    int min_len;    /* minimum length for abbrevation--typically 1    */
    void (*cmd_func) (const char*);  /* function implementing command */
    cmd_flag_t flags;                /* flags for command properties  */
};

static const struct command_t command[] = {
    {"break",     1, cmd_break,     CMD_FLAG_NONE      },
    {"continue",  1, cmd_continue,  CMD_FLAG_REPEATABLE},
    {"dump",      1, cmd_dump,      CMD_FLAG_LIST_TYPE },
    {"execute",   1, cmd_execute,   CMD_FLAG_NONE      },
    {"file",      1, cmd_file,      CMD_FLAG_NONE      },
    {"finish",    3, cmd_finish,    CMD_FLAG_REPEATABLE},
    {"help",      1, cmd_help,      CMD_FLAG_NONE      },
    {"list",      1, cmd_list,      CMD_FLAG_LIST_TYPE },
    {"memory",    1, cmd_memory,    CMD_FLAG_NONE      },
    {"next",      1, cmd_next,      CMD_FLAG_REPEATABLE},
    {"option",    1, cmd_option,    CMD_FLAG_NONE      },
    {"printregs", 1, cmd_printregs, CMD_FLAG_NONE      },
    {"quit",      4, cmd_quit,      CMD_FLAG_NONE      },
    {"register",  1, cmd_register,  CMD_FLAG_NONE      },
    {"reset",     5, cmd_reset,     CMD_FLAG_NONE      },
    {"rstep",	  2, cmd_r_step,    CMD_FLAG_REVERSE | CMD_FLAG_REPEATABLE },
    {"rnext",	  2, cmd_r_next,    CMD_FLAG_REVERSE | CMD_FLAG_REPEATABLE },
    {"rcontinue", 2, cmd_r_continue,CMD_FLAG_REVERSE | CMD_FLAG_REPEATABLE },
    {"rfinish",   2, cmd_r_finish,  CMD_FLAG_REVERSE | CMD_FLAG_REPEATABLE },
    {"step",      1, cmd_step,      CMD_FLAG_REPEATABLE},
    {"translate", 1, cmd_translate, CMD_FLAG_NONE      },
    {"x",         1, cmd_lc3_stop,  CMD_FLAG_GUI_ONLY  },
// commented out until code copied from cmd_breakpoint is written
//    {"watch",     1, cmd_watch,     CMD_FLAG_NONE      },
    {NULL,        0, NULL,          CMD_FLAG_NONE      }
};

static const char* const ccodes[8] = {
    "BAD_CC", "POSITIVE", "ZERO", "BAD_CC",
    "NEGATIVE", "BAD_CC", "BAD_CC", "BAD_CC"
};

static const char* const rname[NUM_REGS + 1] = {
    "R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", 
    "PC", "IR", "PSR", "CC"
};

static const char* const cc_val[4] = {
    "POSITIVE", "ZERO", "", "NEGATIVE"
};

// For reverse execution

// Flags delta variable controls
typedef struct delta_flag_t{
    int valid;
    int step; // The step count for that instruction in the whole program
    int pc; // The PC for that instruction
    cache_type_t command; // see CACHE_TYPE_*
    // (_INST for an instruction, _MEM for memory, _REG for register)
    inst_flag_t inst_flag; // The flag of the instruction. Only valid when current step is an instruction
} delta_flag_t;

// Delta for register change 
typedef struct reg_delta_t{
    int valid;
    reg_num_t reg_num;
    int old_value;
} reg_delta_t;

// Delta for memory change 
typedef struct mem_delta_t{
    int valid;
    int address;
    int old_value;
} mem_delta_t;

// Delta for all possible changes in LC3 for one instuction
typedef struct lc3_delta_t{
    delta_flag_t flags;
    reg_delta_t reg_deltas[MAX_REG_CHANGED];
    mem_delta_t mem_deltas[MAX_MEM_CHANGED];
} lc3_delta_t;

/************************* Global variables for current state *******************/
// Added for reverse execution
static lc3_delta_t cached_step[MAX_CACHED_STEPS]; // cyclic array

// Global counter for how many steps did the program executed
static unsigned int cur_step; 
// Temporary store current delta to previous step
static reg_delta_t cur_reg_delta[MAX_REG_CHANGED]; 
static mem_delta_t cur_mem_delta[MAX_MEM_CHANGED];

static int lc3_register[NUM_REGS];
#define REG(i) lc3_register[(i)]
static int lc3_memory[65536];
static int lc3_show_later[65536];
static bpt_type_t lc3_breakpoints[65536];
static int lc3_bpt_count[65536]; // For skipping a breakpoint until reach the hit times
#ifdef WASM
static int lc3_debug[65536];
#endif // WASM
// commented out until used
//static watch_type_t lc3_watch_mem[65536];
//static watch_type_t lc3_watch_reg[NUM_REGS];

/* startup script or file */
static char* start_script = NULL;
static char* start_file = NULL;

static int should_halt = 1, last_KBSR_read = 0, last_DSR_read = 0, gui_mode;
static int interrupted_at_gui_request = 0, stop_scripts = 0, in_init = 0;
static int have_mem_to_dump = 0, need_a_stop_notice = 0;
static int sys_bpt_addr = -1; // The breakpoint set by system such as "next" 
static int finish_depth = 0; // The depth of subroutine call, used for halt possible infinite recursion
static inst_flag_t last_flags;
/* options and script recursion level */
static int flush_on_start = 1, keep_input_on_stop = 1;
static int rand_device = 1, delay_mem_update = 1;
static int script_uses_stdin = 1, script_depth = 0;

static FILE* lc3in;
static FILE* lc3out; // In gui mode, for the console (printf will be in another window)
static FILE* sim_in;
static char* (*lc3readline) (const char*) = simple_readline;



static int 
execute_instruction ()
{
    // Before every execution, cache PC to the last entry, 
    // IR to last but one entry
    cur_reg_delta[MAX_REG_CHANGED-1].valid = 1; 
    cur_reg_delta[MAX_REG_CHANGED-1].reg_num = R_PC; 
    cur_reg_delta[MAX_REG_CHANGED-1].old_value = REG (R_PC);

    cur_reg_delta[MAX_REG_CHANGED-2].valid = 1; 
    cur_reg_delta[MAX_REG_CHANGED-2].reg_num = R_IR; 
    cur_reg_delta[MAX_REG_CHANGED-2].old_value = REG (R_IR);

    /* Fetch the instruction. */
    REG (R_IR) = read_memory (REG (R_PC));
    REG (R_PC) = (REG (R_PC) + 1) & 0xFFFF;

    /* Try to execute it. */

#define ADD_FLAGS(value) (last_flags |= (value))
#define RECORD_REG_DELTA(i, reg_num_, old_value_) \\
    cur_reg_delta[i].valid = 1;                   \\
    cur_reg_delta[i].reg_num = reg_num_;          \\
    cur_reg_delta[i].old_value = old_value_;
#define RECORD_MEM_DELTA(i, address_)                                     \\
    if (address_ != 0xFE00 && address_ != 0xFE02 && address_ != 0xFE04 && \\
	address_ != 0xFE06 && address_ != 0xFFFE) {                       \\
	cur_mem_delta[i].valid = 1;                                       \\
	cur_mem_delta[i].address = address_;                              \\
	cur_mem_delta[i].old_value = lc3_memory[address_];                \\
    }
#define DEF_INST(name,format,mask,match,flags,code) \\
    if ((REG (R_IR) & (mask)) == (match)) {         \\
	last_flags = (flags);                       \\
	code;     				    \\
	goto executed;                              \\
    }
#define DEF_P_OP(name,format,mask,match)
#include "lc3.def" // The execution is inside this file, using the DEF_INST above
#undef DEF_P_OP
#undef DEF_INST
#undef RECORD_MEM_DELTA
#undef RECORD_REG_DELTA
#undef ADD_FLAGS

    REG (R_PC) = (REG (R_PC) - 1) & 0xFFFF;
    if (gui_mode)
	printf ("ERR {Illegal instruction at x%04X!}\\n", REG (R_PC));
    else
	printf ("Illegal instruction at x%04X!\\n", REG (R_PC));
    return 0;

executed:
    
    // Cache current state 
    cache_ (CACHE_TYPE_INST);

    /* Check for user breakpoints. */
    if (lc3_breakpoints[REG (R_PC)] == BPT_USER) {
	// For skipping breakpoints
	if (0 <= lc3_bpt_count[REG (R_PC)]) {
	    lc3_bpt_count[REG (R_PC)]--;
	}
	if (0 > lc3_bpt_count[REG (R_PC)]) {
	    if (!gui_mode) {
			#ifdef WASM
			EM_ASM_({reportWarn(\$0);}, "The LC-3 hit a breakpoint...\\n");
			#else
			printf ("The LC-3 hit a breakpoint...\\n");
			#endif // WASM
		}
		
	    return 0;
	}
    }

    /* Check for system breakpoint (associated with "next" command). */
    if (REG (R_PC) == sys_bpt_addr)
	return 0;

    // Update depth if it is related to subroutine call/ret
    if (finish_depth > 0) {
	if ((last_flags & FLG_SUBROUTINE) &&
	    ++finish_depth == MAX_FINISH_DEPTH) {
	    if (gui_mode)
		puts ("ERR {Stopping due to possibly infinite recursion.}");
	    else
		puts ("Stopping due to possibly infinite recursion.");
	    finish_depth = 0;
	    return 0;
	} else if ((last_flags & FLG_RETURN) && --finish_depth == 0) {
	    /* Done with finish command; stop execution. */
	    return 0;
	}
    }

    /* Check for GUI needs. */
    if (!in_init && gui_mode) {
	struct pollfd p;

	p.fd = fileno (sim_in);
	p.events = POLLIN;
	if (poll (&p, 1, 0) == 1 && (p.revents & POLLIN) != 0) {
	    interrupted_at_gui_request = 1;
	    return 0;
	}
    }

    return 1;
}


void
halt_lc3 (int sig)
{
    /* Set the signal handler again, which has the effect of overriding
       the Solaris behavior and making signal look like sigset, which
       is non-standard and non-portable, but the desired behavior. */
    signal (SIGINT, halt_lc3);

    /* has no effect unless LC-3 is running... */
    should_halt = 1; // will be read elsewhere to halt lc3

    /* print a stop notice after ^C */
    need_a_stop_notice = 1;
}


static int
launch_gui_connection ()
{
    u_short port;
    int fd;                   /* server socket file descriptor   */
    struct sockaddr_in addr;  /* server socket address           */

    /* wait for the GUI to tell us the portfor the LC-3 console socket */
    if (fscanf (sim_in, "%hd", &port) != 1)
        return -1;

    /* don't buffer output to GUI */
    if (setvbuf (stdout, NULL, _IONBF, 0) == -1)
    	return -1;

    /* create a TCP socket */
    if ((fd = socket (PF_INET, SOCK_STREAM, 0)) == -1)
	return -1;

    /* bind the port to the loopback address with any port */
    bzero (&addr, sizeof (addr));
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl (INADDR_LOOPBACK);
    addr.sin_port        = 0;
    if (bind (fd, (struct sockaddr*)&addr, sizeof (addr)) == -1) {
	close (fd);
	return -1;
    }

    /* now connect to the given port */
    addr.sin_port = htons (port);
    if (connect (fd, (struct sockaddr*)&addr, 
    	         sizeof (struct sockaddr_in)) == -1) {
	close (fd);
	return -1;
    }

    /* use it for LC-3 keyboard and display I/O */
    if ((lc3in = fdopen (fd, "r")) == NULL ||
	(lc3out = fdopen (fd, "w")) == NULL ||
	setvbuf (lc3out, NULL, _IONBF, 0) == -1) {
	close (fd);
        return -1;
    }

    return 0;
}

#ifdef WASM
EM_JS(void, get_input, (char *buffer), {
	// Get the input async
	return Asyncify.handleAsync(async() => {
		await getInput(buffer);
	});
});
#endif // WASM

static char*
simple_readline (const char* prompt)
{
    char buf[200];
    char* strip_nl;
    struct pollfd p;

    /* If we exhaust all commands after being interrupted by the
       GUI, start running again... */
    if (gui_mode) {
	p.fd = fileno (sim_in);
	p.events = POLLIN;
	if ((poll (&p, 1, 0) != 1 || (p.revents & POLLIN) == 0) &&
	    interrupted_at_gui_request) {
	    /* flag is reset to 0 in cmd_continue */
	    return strdup ("c");
	}
    }

    /* Prompt and read a line until successful. */
    while (1) {

#if !defined(USE_READLINE)
	#ifndef WASM
	if (!gui_mode && script_depth == 0)
	    printf ("%s", prompt);
	#endif // WASM
#endif

	#ifdef WASM
	/* We should replace this into js functions */
	get_input(buf);
	break;

	#else
	/* read a line */
	if (fgets (buf, 200, sim_in) != NULL)
	    break;

	/* no more input? */
    	if (feof (sim_in))
	    return NULL;

    	/* Otherwise, probably a CTRL-C, so print a blank line and
	   (possibly) another prompt, then try again. */
    	puts ("");
	#endif // WASM

    }

    /* strip carriage returns and linefeeds */
    for (strip_nl = buf + strlen (buf) - 1;
    	 strip_nl >= buf && (*strip_nl == '\\n' || *strip_nl == '\\r');
	 strip_nl--);
    *++strip_nl = 0;

    return strdup (buf);
}


static void
command_loop ()
{
    int cword_len;
    char* cmd = NULL;
    char* start; // point to the start of the command string
    char* last_cmd = NULL;
    char cword[MAX_CMD_WORD_LEN];
    const command_t* a_command;

    // Has command from script and command line is valid, 
    // or reading from command line input until valid
    while (!stop_scripts && (cmd = lc3readline ("(lc3sim) ")) != NULL) {
	
	/* Skip white space. */
	for (start = cmd; isspace ((unsigned char)*start); start++);

	if (*start == '\\0') { // End of command after skipping space (empty line)
	    /* An empty line repeats the last command, if allowed. */
	    free (cmd); // Current command is empty, useless
	    
	    // If last command is also empty, go the next cycle for new command
	    if ((cmd = last_cmd) == NULL) 
		continue; 

	    /* Skip white space. */
	    for (start = cmd; isspace ((unsigned char)*start); start++);

	} else if (last_cmd != NULL)
	    free (last_cmd); // last command no longer needed to be repeated, become useless
	
	last_cmd = NULL; // either cmd points to the last command, or last command is freed

	// Copy the command pinted by start into cword
	/* Should never fail; just ignore the command if it does. */
	/* 40 below == MAX_CMD_WORD_LEN - 1 */
	if (sscanf (start, "%40s", cword) != 1) {
	    free (cmd);
	    break;
	}

	/* Record command word length, then point to arguments. */
	cword_len = strlen (cword);
	for (start += cword_len; isspace ((unsigned char)*start); start++);
		
	/* Match command word to list of commands. */
	a_command = command; 
	while (1) {
	    if (a_command->command == NULL) {
		/* No match found--complain! */
		free (cmd);
		printf ("Unknown command.  Type 'h' for help.\\n");
		break;
	    }

	    /* Try to match a_command. */
	    if (strncasecmp (cword, a_command->command, cword_len) == 0 &&
		cword_len >= a_command->min_len &&
		(gui_mode || (a_command->flags & CMD_FLAG_GUI_ONLY) == 0)) {

		/* Execute the command. */
		(*a_command->cmd_func) (start);

		/* Handle list type and repeatable commands. */
		if (a_command->flags & CMD_FLAG_LIST_TYPE) {
		    char buf[MAX_CMD_WORD_LEN + 5];

		    strcpy (buf, cword);
		    strcat (buf, " more");
		    last_cmd = strdup (buf);
		} else if (a_command->flags & CMD_FLAG_REPEATABLE &&
			   script_depth == 0) {
		    last_cmd = cmd; // For executing the same command just by pressing enter
		} else {
		    free (cmd);
		}
		break;
	    }
	    
	    // This is for matching a command from the list, until it go into one of the if above
	    a_command++; 
	}
    }
}

#ifdef WASM
int
main_lc3sim (char* input)
#else
int
main (int argc, char** argv)
#endif // WASM
{
	#ifdef WASM
	int argc = 1;
	char **argv;
	#endif // WASM
    /* check for -gui argument */
    sim_in = stdin;
    if (argc > 1 && strcmp (argv[1], "-gui") == 0) {
    	if (launch_gui_connection () != 0) {
	    printf ("failed to connect to GUI\\n");
	    return 1;
	}
	/* skip the -gui argument in later parsing */
	argc--;
	argv++;
	gui_mode = 1;
    } else {
        lc3out = stdout;
	lc3in = stdin;
	gui_mode = 0;
    }

    /* used to simulate random device timing behavior */
    srandom (time (NULL)); // used for randomize KBSR and DSR

	#ifdef WASM
	start_file = strdup(input);
	init_machine(); /* also loads file */
	command_loop();
	#else
    /* used to halt LC-3 when CTRL-C pressed */
    signal (SIGINT, halt_lc3);

    /* load any object, symbol, or script files requested on command line */
    if (argc == 3 && strcmp (argv[1], "-s") == 0) {
	start_script = argv[2];
	init_machine (); /* also executes script */
	return 0;
    } else if (argc == 2 && strcmp (argv[1], "-h") != 0) {
	start_file = strdup (argv[1]);
	init_machine (); /* also loads file */
    } else if (argc != 1) {
	/* argv[0] may not be valid if -gui entered */
	printf ("syntax: lc3sim [<object file>|<symbol file>]\\n");
	printf ("        lc3sim [-s <script file>]\\n");
	printf ("        lc3sim -h\\n");
	return 0;
    } else
    	init_machine ();

    command_loop ();

	#endif // WASM

	puts ("");
    return 0;
}

int 
read_memory (int addr)
{
    struct pollfd p;

    switch (addr) {
	case 0xFE00: /* KBSR */
	    if (!last_KBSR_read) {
	        p.fd = fileno (lc3in);
		p.events = POLLIN;
		if (poll (&p, 1, 0) == 1 && (p.revents & POLLIN) != 0)
		    last_KBSR_read = (!rand_device || (random () & 15) == 0);
	    }
	    return (last_KBSR_read ? 0x8000 : 0x0000);
	case 0xFE02: /* KBDR */
	    if (last_KBSR_read && (lc3_memory[0xFE02] = fgetc (lc3in)) == -1) {
	    	/* Should not happen in GUI mode. */
		/* FIXME: This won't show up correctly in GUI.
		   Exit is likely to be detected first, and error message
		   given (LC-3 sim. died), followed by message below 
		   (read past end), then Tcl/Tk error caused by bad
		   window access after sim died.  Confusing sequence
		   if it occurs. */
		if (gui_mode)
		    puts ("ERR {LC-3 read past end of input stream.}");
	    	else
		    puts ("LC-3 read past end of input stream.");
		exit (3);
	    }
	    last_KBSR_read = 0;
	    return lc3_memory[0xFE02];
	case 0xFE04: /* DSR */
	    if (!last_DSR_read)
	        last_DSR_read = (!rand_device || (random () & 15) == 0);
	    return (last_DSR_read ? 0x8000 : 0x0000);
	case 0xFE06: /* DDR */
	    return 0x0000;
	case 0xFFFE: return 0x8000;   /* MCR */
    }
    return lc3_memory[addr];
}

void 
write_memory (int addr, int value)
{
    switch (addr) {
	case 0xFE00: /* KBSR */
	case 0xFE02: /* KBDR */
	case 0xFE04: /* DSR */
	    return;
	case 0xFE06: /* DDR */
	    if (last_DSR_read == 0)
	    	return;
	    #ifdef WASM
	    EM_ASM_({
		    reportDDR(\$0);
	    },value);
	    #else
	    fprintf (lc3out, "%c", value);
	    #endif // WASM
	    fflush (lc3out);
	    last_DSR_read = 0;
	    return;
	case 0xFFFE:
	    if ((value & 0x8000) == 0)
	    	should_halt = 1;
	    return;
    }
    /* No need to write/update GUI if the same value is already in memory. */
    if (value != lc3_memory[addr]) {
	lc3_memory[addr] = value;
	if (gui_mode) {
	    if (!delay_mem_update)
		#ifdef WASM
		disassemble_one (addr, 0);
		#else
		disassemble_one (addr);
		#endif // WASM
	    else {
		lc3_show_later[addr] = 1;
		have_mem_to_dump = 1; /* a hint */
	    }
	}
    }
}

#ifdef WASM
static int
read_obj_file (const unsigned char* filename, const unsigned char* debugname, int* startp, int* endp)
{
    FILE* f;
	FILE *debug = NULL;
	int start, addr;
	int linenum;
	unsigned char buf[2];

	if ((f = fopen (filename, "r")) == NULL)
		return -1;

	if (debugname != NULL) {
		debug = fopen(debugname, "r");
	}

	if (fread (buf, 2, 1, f) != 1) {
        fclose (f);
	return -1;
    }
	if (debug != NULL) { // read ORIG and ignore
		printf("Loaded debug file \\"%s\\"\\n", debugname);
		fscanf(debug, "%d", &linenum);
	}
	addr = start = (buf[0] << 8) | buf[1];
	while (fread (buf, 2, 1, f) == 1) {
		if (debug != NULL) {
			fscanf(debug, "%d", &linenum);
			lc3_debug[addr] = linenum;
		}
	write_memory (addr, (buf[0] << 8) | buf[1]);
	addr = (addr + 1) & 0xFFFF;
    }
    fclose (f);
	if(debug != NULL)
		fclose(debug);
	squash_symbols(start, addr);
	*startp = start;
    *endp = addr;

	if(debug != NULL) {
		EM_ASM_({
			setDebugInfo(\$0);
		}, lc3_debug);
	}

    return 0;
}
#else
static int
read_obj_file (const char* filename, int* startp, int* endp)
{
    FILE* f;
    int start, addr;
    unsigned char buf[2];

    if ((f = fopen (filename, "r")) == NULL)
	return -1;
    if (fread (buf, 2, 1, f) != 1) {
        fclose (f);
	return -1;
    }
    addr = start = (buf[0] << 8) | buf[1];
    while (fread (buf, 2, 1, f) == 1) {
	write_memory (addr, (buf[0] << 8) | buf[1]);
	addr = (addr + 1) & 0xFFFF;
    }
    fclose (f);
    squash_symbols (start, addr);
    *startp = start;
    *endp = addr;

    return 0;
}
#endif // WASM

static int
read_sym_file (const char* filename)
{
    FILE* f;
    int adding = 0;
    char buf[100];
    char sym[81];
    int addr;

    if ((f = fopen (filename, "r")) == NULL)
	return -1;
    while (fgets (buf, 100, f) != NULL) {
    	if (!adding) {
	    if (sscanf (buf, "%*s%*s%80s", sym) == 1 &&
	    	strcmp (sym, "------------") == 0)
		adding = 1;
	    continue;
	}
	if (sscanf (buf, "%*s%80s%x", sym, &addr) != 2)
	    break;
        add_symbol (sym, addr, 1);
    }
    fclose (f);
    return 0;
}


static void 
squash_symbols (int addr_s, int addr_e)
{
    while (addr_s != addr_e) {
	remove_symbol_at_addr (addr_s);
	addr_s = (addr_s + 1) & 0xFFFF;
    }
}


static void 
init_machine ()
{
    int os_start, os_end;

    in_init = 1;

    bzero (lc3_register, sizeof (lc3_register));
    REG (R_PSR) = (2L << 9); /* set to condition ZERO */
    bzero (lc3_memory, sizeof (lc3_memory));
    bzero (lc3_show_later, sizeof (lc3_show_later));
    bzero (lc3_sym_names, sizeof (lc3_sym_names));
    bzero (lc3_sym_hash, sizeof (lc3_sym_hash));

    // Initialize cache before executing anything; will be reinitialized
    // by OS load, but only if that code is available.
    clear_cache ();

    clear_all_breakpoints ();

	#ifdef WASM
	if (read_obj_file ("./lc3os.obj", NULL, &os_start, &os_end) == -1) {
	#else
	if (read_obj_file ( INSTALL_DIR "/lc3os.obj", &os_start, &os_end) == -1) {
	#endif // WASM
	if (gui_mode)
	    puts ("ERR {Failed to read LC-3 OS code.}");
	else
	    puts ("Failed to read LC-3 OS code.");
	show_state_if_stop_visible ();
    } else {
	#ifdef WASM
	if (read_sym_file ("./lc3os.sym") == -1) {
	#else
	if (read_sym_file (INSTALL_DIR "/lc3os.sym") == -1) {
	#endif // WASM
	    if (gui_mode)
		puts ("ERR {Failed to read LC-3 OS symbols.}");
	    else
		puts ("Failed to read LC-3 OS symbols.");
	}
	if (gui_mode) /* load new code into GUI display */
	    disassemble (os_start, os_end);
	REG (R_PC) = 0x0200;
	run_until_stopped ();
    }

    in_init = 0;

    if (start_script != NULL) 
	cmd_execute (start_script);
    else if (start_file != NULL) {
		#ifdef WASM
		char *buf = strdup(start_file);
		char *ch;
		ch = strtok(buf, " ");
		while (ch != NULL)
		{
			// printf("%s\\n", ch);
			cmd_file(ch);
			ch = strtok(NULL, " ,");
		}
		#else
		cmd_file (start_file);
		#endif // WASM
		
	}
	
}


/* only called in GUI mode */

static void
print_register (int which)
{
    printf ("REG R%d x%04X\\n", which, REG (which));
    /* condition codes are not stored outside of PSR */
    if (which == R_PSR)
	printf ("REG R%d %s\\n", NUM_REGS, ccodes[(REG (R_PSR) >> 9) & 7]);
    /* change focus in GUI */
    printf ("TOCODE\\n");
}

static void
print_registers ()
{
    int regnum;

    if (!gui_mode) {
		#ifdef WASM
		disassemble_one (REG (R_PC), 1);
		EM_ASM_({
			setRegisters(\$0);
		},lc3_register);
		EM_ASM_({
			setLineNum(\$0);
		},lc3_debug[REG(R_PC)]);
		#else
		printf ("PC=x%04X IR=x%04X PSR=x%04X (%s)\\n", REG (R_PC), REG (R_IR),
			REG (R_PSR), ccodes[(REG (R_PSR) >> 9) & 7]);
		for (regnum = 0; regnum < R_PC; regnum++)
			printf ("R%d=x%04X ", regnum, REG (regnum));
		puts ("");
		disassemble_one (REG (R_PC));
		#endif // WASM
    } else {
	for (regnum = 0; regnum < NUM_REGS; regnum++)
	    printf ("REG R%d x%04X\\n", regnum, REG (regnum));
	/* regnum is now NUM_REGS */
    	printf ("REG R%d %s\\n", regnum, ccodes[(REG (R_PSR) >> 9) & 7]);
    }
}

#ifdef WASM
int stringLen(char *p) {
	return strlen(p);
}
#endif // WASM

static void 
dump_delayed_mem_updates ()
{
    int addr;

    if (!have_mem_to_dump)
        return;
    have_mem_to_dump = 0;

    /* FIXME: Could use a hash table here, but hint is probably enough. */
    for (addr = 0; addr < 65536; addr++) {
        if (lc3_show_later[addr]) {
			#ifdef WASM
			disassemble_one (addr, 0);
			#else
			disassemble_one (addr);
			#endif // WASM
	    lc3_show_later[addr] = 0;
	}
    }
}


static void
show_state_if_stop_visible ()
{
    /* 
       If the GUI has interrupted the simulator (e.g., to set or clear
       a breakpoint), print nothing.  The simulator restarts automatically
       unless a new file is loaded, in which case cmd_file performs the
       updates. 
    */
    if (interrupted_at_gui_request)
        return;

    if (gui_mode && delay_mem_update)
	dump_delayed_mem_updates ();
    print_registers ();
}

#ifdef WASM
static void
print_operands (char* operands, int addr, int inst, format_t fmt)
{
	int found = 0, tgt;

	if (fmt & FMT_R1) {
    	sprintf (operands, "%s%sR%d", operands, (found ? "," : ""), F_DR (inst));
	found = 1;
    }
    if (fmt & FMT_R2) {
		sprintf(operands, "%s%sR%d", operands,(found ? "," : ""), F_SR1(inst));
		found = 1;
    }
    if (fmt & FMT_R3) {
		sprintf(operands, "%s%sR%d", operands,(found ? "," : ""), F_SR2(inst));
		found = 1;
    }
    if (fmt & FMT_IMM5) {
		sprintf(operands, "%s%s#%d", operands,(found ? "," : ""), F_imm5(inst));
		found = 1;
    }
    if (fmt & FMT_IMM6) {
		sprintf(operands, "%s%s#%d", operands,(found ? "," : ""), F_imm6(inst));
		found = 1;
    }
    if (fmt & FMT_VEC8) {
		sprintf(operands, "%s%sx%02X", operands,(found ? "," : ""), F_vec8(inst));
		found = 1;
    }
    if (fmt & FMT_ASC8) {
    	sprintf (operands, "%s%s",operands, (found ? "," : ""));
	found = 1;
	switch (F_vec8 (inst)) {
	    case  7:
			sprintf(operands, "%s'\\\\a'", operands);
			break;
		case  8:
			sprintf(operands, "%s'\\\\b'", operands);
			break;
		case  9:
			sprintf(operands, "%s'\\\\t'", operands);
			break;
		case 10:
			sprintf(operands, "%s'\\\\n'", operands);
			break;
		case 11: sprintf (operands, "%s'\\\\v'", operands); break;
	    case 12: sprintf (operands, "%s'\\\\f'", operands); break;
	    case 13: sprintf (operands, "%s'\\\\r'", operands); break;
	    case 27: sprintf (operands, "%s'\\\\e'", operands); break;
	    case 34: sprintf (operands, "%s'\\\\\\"'", operands); break;
	    case 44: sprintf (operands, "%s'\\\\''", operands); break;
	    case 92: sprintf (operands, "%s'\\\\\\\\'", operands); break;
	    default:
	    	if (isprint (F_vec8 (inst)))
		    sprintf (operands, "%s'%c'", operands,F_vec8 (inst));
		else
			sprintf(operands, "%sx%02X", operands, F_vec8(inst));
		break;
	}
    }
    if (fmt & FMT_IMM9) {
    	sprintf (operands, "%s%s", operands, (found ? "," : ""));
	found = 1;
	tgt = (addr + 1 + F_imm9 (inst)) & 0xFFFF;
	if (lc3_sym_names[tgt] != NULL)
	    sprintf (operands, "%s%s", operands, lc3_sym_names[tgt]->name);
    	else
			sprintf(operands, "%sx%04X", operands, tgt);
	}
    if (fmt & FMT_IMM11) {
		sprintf(operands, "%s%s", operands,(found ? "," : ""));
		found = 1;
		tgt = (addr + 1 + F_imm11(inst)) & 0xFFFF;
		if (lc3_sym_names[tgt] != NULL)
			sprintf(operands, "%s%s", operands, lc3_sym_names[tgt]->name);
		else
			sprintf(operands, "%sx%04X", operands, tgt);
	}
    if (fmt & FMT_IMM16) {
		sprintf(operands, "%s%s", operands,(found ? "," : ""));
		found = 1;
		if (lc3_sym_names[inst] != NULL)
			sprintf(operands, "%s%s", operands, lc3_sym_names[inst]->name);
		else
	    sprintf (operands, "%sx%04X",operands, inst);
    }
}

int disassemble_one_export (int addr, char* label, char* op, char* operands)
{

    static const char* const dis_cc[8] = {
        "", "P", "Z", "ZP", "N", "NP", "NZ", "NZP"
    };
    int inst = read_memory (addr);

	label[0] = 0;
	op[0] = 0;
	operands[0] = 0;

	/* Try to find a label. */
	if (lc3_sym_names[addr] != NULL)
		sprintf(label, "%c %16.16s x%04X x%04X ",
				(lc3_breakpoints[addr] == BPT_USER ? 'B' : ' '),
				lc3_sym_names[addr]->name, addr, inst);
	else
		sprintf(label, "%c %17sx%04X x%04X ",
				(lc3_breakpoints[addr] == BPT_USER ? 'B' : ' '),
				"", addr, inst);

	int cnt = 0;

	/* Try to disassemble it. */

#define DEF_INST(name, format, mask, match, flags, code)                      \\
	if ((inst & (mask)) == (match))                                           \\
	{                                                                         \\
		if ((format)&FMT_CC)                                                  \\
			sprintf(op, "%s%-*s", #name, (int)(OPCODE_WIDTH - strlen(#name)), \\
					dis_cc[F_CC(inst) >> 9]);                                 \\
		else                                                                  \\
			sprintf(op, "%-*s", OPCODE_WIDTH, #name);                         \\
		print_operands(operands, addr, inst, (format));                       \\
		goto printed;                                                         \\
	}
#define DEF_P_OP(name,format,mask,match) \\
    DEF_INST(name,format,mask,match,FLG_NONE,{})
#include "lc3.def"
#undef DEF_P_OP
#undef DEF_INST

    sprintf (label, "%-*s", OPCODE_WIDTH, "???");

printed:
	// printf("operands: %s\\n", operands);
	return inst;
}

static void 
disassemble_one (int addr, int should_report)
{

    static const char* const dis_cc[8] = {
        "", "P", "Z", "ZP", "N", "NP", "NZ", "NZP"
    };
    int inst = read_memory (addr);

    /* GUI prefix */
    if (gui_mode)
    	printf ("CODE%c%5d", 
	        (!in_init && addr == lc3_register[R_PC] ? 'P' : ' '),
		addr + 1);

	char label[100] = "";	// assume label is less than 200
	char op[100] = "";
	char operands[100] = "";

	// memset(label, 0, 100);
	// memset(op, 0, 100);
	// memset(operands, 0, 100);

	/* Try to find a label. */
	if (lc3_sym_names[addr] != NULL)
		sprintf(label, "%c %16.16s x%04X x%04X ",
				(lc3_breakpoints[addr] == BPT_USER ? 'B' : ' '),
				lc3_sym_names[addr]->name, addr, inst);
	else
		sprintf(label, "%c %17sx%04X x%04X ",
				(lc3_breakpoints[addr] == BPT_USER ? 'B' : ' '),
				"", addr, inst);

	int cnt = 0;

	/* Try to disassemble it. */

#define DEF_INST(name, format, mask, match, flags, code)                      \\
	if ((inst & (mask)) == (match))                                           \\
	{                                                                         \\
		if ((format)&FMT_CC)                                                  \\
			sprintf(op, "%s%-*s", #name, (int)(OPCODE_WIDTH - strlen(#name)), \\
					dis_cc[F_CC(inst) >> 9]);                                 \\
		else                                                                  \\
			sprintf(op, "%-*s", OPCODE_WIDTH, #name);                         \\
		print_operands(operands, addr, inst, (format));                       \\
		goto printed;                                                         \\
	}
#define DEF_P_OP(name,format,mask,match) \\
    DEF_INST(name,format,mask,match,FLG_NONE,{})
#include "lc3.def"
#undef DEF_P_OP
#undef DEF_INST

    sprintf (label, "%-*s", OPCODE_WIDTH, "???");

printed:
    if(should_report) {
		// printf("label: %s, op: %s, operands: %s\\n", label, op, operands);
		EM_ASM_({
			setLineInfo(\$0, \$1, \$2);
		},
				label, op, operands);
	}
}
#else
static void
print_operands (int addr, int inst, format_t fmt)
{
    int found = 0, tgt;

    if (fmt & FMT_R1) {
    	printf ("%sR%d", (found ? "," : ""), F_DR (inst));
	found = 1;
    }
    if (fmt & FMT_R2) {
    	printf ("%sR%d", (found ? "," : ""), F_SR1 (inst));
	found = 1;
    }
    if (fmt & FMT_R3) {
    	printf ("%sR%d", (found ? "," : ""), F_SR2 (inst));
	found = 1;
    }
    if (fmt & FMT_IMM5) {
    	printf ("%s#%d", (found ? "," : ""), F_imm5 (inst));
	found = 1;
    }
    if (fmt & FMT_IMM6) {
    	printf ("%s#%d", (found ? "," : ""), F_imm6 (inst));
	found = 1;
    }
    if (fmt & FMT_VEC8) {
    	printf ("%sx%02X", (found ? "," : ""), F_vec8 (inst));
	found = 1;
    }
    if (fmt & FMT_ASC8) {
    	printf ("%s", (found ? "," : ""));
	found = 1;
	switch (F_vec8 (inst)) {
	    case  7: printf ("'\\\\a'"); break;
	    case  8: printf ("'\\\\b'"); break;
	    case  9: printf ("'\\\\t'"); break;
	    case 10: printf ("'\\\\n'"); break;
	    case 11: printf ("'\\\\v'"); break;
	    case 12: printf ("'\\\\f'"); break;
	    case 13: printf ("'\\\\r'"); break;
	    case 27: printf ("'\\\\e'"); break;
	    case 34: printf ("'\\\\\\"'"); break;
	    case 44: printf ("'\\\\''"); break;
	    case 92: printf ("'\\\\\\\\'"); break;
	    default:
	    	if (isprint (F_vec8 (inst)))
		    printf ("'%c'", F_vec8 (inst));
		else
		    printf ("x%02X", F_vec8 (inst));
		break;
	}
    }
    if (fmt & FMT_IMM9) {
    	printf ("%s", (found ? "," : ""));
	found = 1;
	tgt = (addr + 1 + F_imm9 (inst)) & 0xFFFF;
	if (lc3_sym_names[tgt] != NULL)
	    printf ("%s", lc3_sym_names[tgt]->name);
    	else
	    printf ("x%04X", tgt);
    }
    if (fmt & FMT_IMM11) {
    	printf ("%s", (found ? "," : ""));
	found = 1;
	tgt = (addr + 1 + F_imm11 (inst)) & 0xFFFF;
	if (lc3_sym_names[tgt] != NULL)
	    printf ("%s", lc3_sym_names[tgt]->name);
    	else
	    printf ("x%04X", tgt);
    }
    if (fmt & FMT_IMM16) {
    	printf ("%s", (found ? "," : ""));
	found = 1;
	if (lc3_sym_names[inst] != NULL)
	    printf ("%s", lc3_sym_names[inst]->name);
    	else
	    printf ("x%04X", inst);
    }
}

static void 
disassemble_one (int addr)
{
    static const char* const dis_cc[8] = {
        "", "P", "Z", "ZP", "N", "NP", "NZ", "NZP"
    };
    int inst = read_memory (addr);

    /* GUI prefix */
    if (gui_mode)
    	printf ("CODE%c%5d", 
	        (!in_init && addr == lc3_register[R_PC] ? 'P' : ' '),
		addr + 1);
      
    /* Try to find a label. */
    if (lc3_sym_names[addr] != NULL)
	printf ("%c %16.16s x%04X x%04X ", 
		(lc3_breakpoints[addr] == BPT_USER ? 'B' : ' '),
		lc3_sym_names[addr]->name, addr, inst);
    else
	printf ("%c %17sx%04X x%04X ", 
		(lc3_breakpoints[addr] == BPT_USER ? 'B' : ' '),
		"", addr, inst);

    /* Try to disassemble it. */

#define DEF_INST(name,format,mask,match,flags,code)                        \\
    if ((inst & (mask)) == (match)) {                                      \\
	if ((format) & FMT_CC)                                             \\
	    printf ("%s%-*s", #name, (int)(OPCODE_WIDTH - strlen (#name)), \\
	    	    dis_cc[F_CC (inst) >> 9]);                             \\
	else                                                               \\
	    printf ("%-*s", OPCODE_WIDTH, #name);                          \\
	print_operands (addr, inst, (format));			           \\
	goto printed;                                                      \\
    }
#define DEF_P_OP(name,format,mask,match) \\
    DEF_INST(name,format,mask,match,FLG_NONE,{})
#include "lc3.def"
#undef DEF_P_OP
#undef DEF_INST

    printf ("%-*s", OPCODE_WIDTH, "???");

printed:
    puts ("");
}
#endif // WASM

static void 
disassemble (int addr_s, int addr_e)
{
    do {
		#ifdef WASM
		disassemble_one (addr_s, 0);
		#else
		disassemble_one (addr_s);
		#endif // WASM
	addr_s = (addr_s + 1) & 0xFFFF;
    } while (addr_s != addr_e);
}


static void
dump_memory (int addr_s, int addr_e)
{
    int start, addr, i;
    int a[12];

    if (addr_s >= addr_e)
        addr_e += 0x10000;
    for (start = (addr_s / 12) * 12; start < addr_e; start = start + 12) {
        printf ("%04X: ", start & 0xFFFF);
	for (i = 0, addr = start; i < 12; i++, addr++) {
	    if (addr >= addr_s && addr < addr_e)
	        printf ("%04X ", (a[i] = read_memory (addr & 0xFFFF)));
	    else
	        printf ("     ");
	}
	printf (" ");
	for (i = 0, addr = start; i < 12; i++, addr++) {
	    if (addr >= addr_s && addr < addr_e)
	        printf ("%c", (a[i] < 0x100 && isprint (a[i])) ? a[i] : '.');
	    else
	        printf (" ");
	}
	puts ("");
    }
}


static void
run_until_stopped ()
{
    struct termios tio;
    int old_lflag, old_min, old_time, tty_fail;

    should_halt = 0;
    if (gui_mode) {
	/* removes PC marker in GUI */
	printf ("CONT\\n");
        tty_fail = 1;
    } else if (!isatty (fileno (lc3in)) || 
    	       tcgetattr (fileno (lc3in), &tio) != 0)
        tty_fail = 1;
    else {
        tty_fail = 0;
	old_lflag = tio.c_lflag;
	old_min = tio.c_cc[VMIN];
	old_time = tio.c_cc[VTIME];
	tio.c_lflag &= ~(ICANON | ECHO);
	tio.c_cc[VMIN] = 1;
	tio.c_cc[VTIME] = 0;
	(void)tcsetattr (fileno (lc3in), TCSANOW, &tio);
    }

    while (!should_halt && execute_instruction ());

    if (!tty_fail) {
	tio.c_lflag = old_lflag;
	tio.c_cc[VMIN] = old_min;
	tio.c_cc[VTIME] = old_time;
	(void)tcsetattr (fileno (lc3in), TCSANOW, &tio);
	/* 
	   Discard any remaining input if requested.  This flush occurs
	   when the LC-3 stops, in which case any remaining input
	   to the console will be treated as simulator commands if it
	   is not discarded.

	   However, discarding can interfere with command sequences that 
	   include moderately long execution periods.

	   As with gdb, not discarding is the default, since typing in
	   a bunch of random junk that happens to look like valid
	   commands happens less frequently than the case above, although
	   I myself have been bitten a few times in gdb by pressing
	   return once too often after issuing a repeatable command.
	*/
	if (!keep_input_on_stop)
	    (void)tcflush (fileno (lc3in), TCIFLUSH);
    }

    /* stopped by CTRL-C?  Check if we need a stop notice... */
    if (need_a_stop_notice) {
        printf ("\\nLC-3 stopped.\\n\\n");
	need_a_stop_notice = 0;
    }

    /* 
       If stopped for any reason other than interruption by GUI,
       clear system breakpoint and terminate any "finish" command.
    */
    if (!interrupted_at_gui_request) {
	sys_bpt_addr = -1;
	finish_depth = 0;
    }

    /* Dump memory and registers if necessary. */
    show_state_if_stop_visible ();
}


static void
clear_breakpoint (int addr)
{
    if (lc3_breakpoints[addr] != BPT_USER) {
	if (!gui_mode)
		#ifdef WASM
		EM_ASM_({reportErr(\$0);}, "No such breakpoint was set.\\n");
		#else
		printf ("No such breakpoint was set.\\n");
		#endif // WASM
    } else {
	if (gui_mode)
	    printf ("BCLEAR %d\\n", addr + 1);
	else {
		#ifdef WASM
		char msg[40];
		sprintf(msg, "Cleared breakpoint at x%04X.\\n", addr);
		EM_ASM_({
			reportSucc(\$0);
		}, msg);
		#else
		printf ("Cleared breakpoint at x%04X.\\n", addr);
		#endif // WASM
	}
	    
    }
    lc3_breakpoints[addr] = BPT_NONE;
}


static void
clear_all_breakpoints ()
{
    /* 
       If other breakpoint types are to be supported,
       this code needs to avoid clobbering non-user
       breakpoints.
    */
    bzero (lc3_breakpoints, sizeof (lc3_breakpoints));
}


static void
list_breakpoints ()
{
    int i, found = 0;

    /* A bit hokey, but no big deal for this few. */
    for (i = 0; i < 65536; i++) {
    	if (lc3_breakpoints[i] == BPT_USER) {
	    if (!found) {
		printf ("The following instructions are set as "
			"breakpoints:\\n");
		found = 1;
	    }
		#ifdef WASM
		disassemble_one (i, 0);
		#else
		disassemble_one (i);
		#endif // WASM
	}
    }

    if (!found)
    	printf ("No breakpoints are set.\\n");
}


static void
set_breakpoint (int addr, int skip_count)
{
    if (lc3_breakpoints[addr] == BPT_USER) {
	if (lc3_bpt_count[addr] != skip_count) {
	    lc3_bpt_count[addr] = skip_count;
	    if (!gui_mode) {
			#ifdef WASM
			char msg[80];
			sprintf(msg, "Skip count set to %d for breakpoint at x%04X.\\n", skip_count, addr);
			EM_ASM_({
				reportWarn(\$0);
			}, msg);
			#else
			printf ("Skip count set to %d for breakpoint at x%04X.\\n", skip_count, addr);
			#endif // WASM
		}
		
	} else if (!gui_mode) {
		#ifdef WASM
		EM_ASM_({reportErr(\$0);}, "That breakpoint is already set.\\n");
		#else
		printf ("That breakpoint is already set.\\n");
		#endif // WASM
	}
    } else {
	lc3_breakpoints[addr] = BPT_USER;
	lc3_bpt_count[addr] = skip_count;
	if (gui_mode)
	    printf ("BREAK %d\\n", addr + 1);
	else if (0 < skip_count) {
		#ifdef WASM
		char msg[80];
		sprintf(msg, "Set breakpoint at x%04X with skip count %d.\\n", addr, skip_count);
		EM_ASM_({
			reportSucc(\$0);
		}, msg);
		#else
		printf ("Set breakpoint at x%04X with skip count %d.\\n", addr, skip_count);
		#endif // WASM
	} else {
		#ifdef WASM
		char msg[40];
		sprintf(msg, "Set breakpoint at x%04X.\\n", addr);
		EM_ASM_({
			reportSucc(\$0);
		}, msg);
		#else
		printf ("Set breakpoint at x%04X.\\n", addr);
		#endif // WASM
	}
    }
}

/**************************** cmd instructions ***************************/

static void 
cmd_break (const char* args)
{
    char opt[11], addr_str[MAX_LABEL_LEN];
    char skip_str[MAX_LABEL_LEN], trash[2];
    int num_args, opt_len, addr, skip_cnt;

    /* 80 == MAX_LABEL_LEN - 1 */
    num_args = sscanf (args, "%10s%80s%80s%1s", opt, addr_str, 
		       skip_str, trash);

    if (num_args > 0) {
	opt_len = strlen (opt);
	if (strncasecmp (opt, "list", opt_len) == 0) {
	    if (num_args > 1)
		warn_too_many_args ();
	    list_breakpoints ();
	    return;
	}
	if (num_args > 1) {
	    if (num_args > 3)
		warn_too_many_args ();
	    addr = parse_address (addr_str);
	    if (num_args > 2) {
	        skip_cnt = parse_address (skip_str);
		if (-1 == skip_cnt) {
		    puts (BAD_ADDRESS);
		    return;
		}
	    } else {
	        skip_cnt = 0;
	    }
	    if (strncasecmp (opt, "clear", opt_len) == 0) {
		if (num_args > 2) {
		    warn_too_many_args ();
		}
		if (strcasecmp (addr_str, "all") == 0) {
		    clear_all_breakpoints ();
		    if (!gui_mode)
			printf ("Cleared all breakpoints.\\n");
		    return;
		}
		if (addr != -1)
		    clear_breakpoint (addr);
		else
		    puts (BAD_ADDRESS);
		return;
	    } else if (strncasecmp (opt, "set", opt_len) == 0) {
		if (addr != -1)
		    set_breakpoint (addr, skip_cnt);
		else
		    puts (BAD_ADDRESS);
		return;
	    }
	}
    }

    printf ("breakpoint options include:\\n");
    printf ("  break clear <addr>|all    -- clear one or all breakpoints\\n");
    printf ("  break list                -- list all breakpoints\\n");
    printf ("  break set <addr> [<skip>] -- set a breakpoint\\n");
}


static void
warn_too_many_args ()
{
    /* Spaces in entry boxes in the GUI appear as
       extra arguments when handed to the command line;
       we silently ignore them. */
    if (!gui_mode)
        puts (TOO_MANY_ARGS);
}


static void
no_args_allowed (const char* args)
{
    if (*args != '\\0')
        warn_too_many_args ();
}


static void
cmd_continue (const char* args)
{
    no_args_allowed (args);
    if (interrupted_at_gui_request)
	interrupted_at_gui_request = 0;
    else
	flush_console_input ();
    run_until_stopped ();
}


static void 
cmd_dump (const char* args)
{
    static int last_end = 0;
    int start, end;

    if (parse_range (args, &start, &end, last_end, 48) == 0) {
	dump_memory (start, end);
	last_end = end;
	return;
    }

    printf ("dump options include:\\n");
    printf ("  dump               -- dump memory around PC\\n");
    printf ("  dump <addr>        -- dump memory starting from an "
    	    "address or label\\n");
    printf ("  dump <addr> <addr> -- dump a range of memory\\n");
    printf ("  dump more          -- continue previous dump (or press "
	    "<Enter>)\\n");
}


static void
cmd_execute (const char* args)
{
    FILE* previous_input;
    FILE* script;

    if (script_depth == MAX_SCRIPT_DEPTH) {
	/* Safer to exit than to bury a warning arbitrarily deep. */
	printf ("Cannot execute more than %d levels of scripts!\\n", 
		MAX_SCRIPT_DEPTH);
	stop_scripts = 1;
	return;
    }

    if ((script = fopen (args, "r")) == NULL) {
        printf ("Cannot open script file \\"%s\\".\\n", args);
	stop_scripts = 1;
	return;
    }

    script_depth++;
    previous_input = sim_in;
    sim_in = script;
    if (!script_uses_stdin)
	lc3in = script;
#if defined(USE_READLINE)
    lc3readline = simple_readline;
#endif
    command_loop ();
    sim_in = previous_input;
    if (--script_depth == 0) {
	if (gui_mode) {
	    lc3in = lc3out;
	} else {
	    lc3in = stdin;
	}
    	stop_scripts = 0;
    } else if (!script_uses_stdin) {
	/* executing previous script level--take LC-3 console input 
	   from script */
	lc3in = previous_input;
    }
    fclose (script);
}


static void
cmd_file (const char* args)
{
    /* extra 4 chars in buf for ".obj" possibly added later */ 
    char buf[MAX_FILE_NAME_LEN + 4];
	#ifdef WASM
	unsigned char bufdebug[MAX_FILE_NAME_LEN + 4];
	#endif // WASM
    char* ext;
    int len, start, end, warn = 0;

    len = strlen (args);
    if (len == 0 || len > MAX_FILE_NAME_LEN - 1) {
	if (gui_mode)
	    printf ("ERR {Could not parse file name!}\\n");
	else
	    printf ("syntax: file <file to load>\\n");
	return;
    }
    strcpy (buf, args);
	#ifdef WASM
	strcpy (bufdebug, args);
	#endif // WASM
    /* FIXME: Need to use portable path element separator characters
       rather than assuming use of '/'. */
    if ((ext = strrchr (buf, '.')) == NULL || strchr (ext, '/') != NULL) {
	ext = buf + len;
        strcat (buf, ".obj");
		#ifdef WASM
		strcat (bufdebug, ".debug");
		#endif // WASM
    } else {
	if (!gui_mode && strcasecmp (ext, ".sym") == 0) {
	    if (read_sym_file (buf))
		printf ("Failed to read symbols from \\"%s.\\"\\n", buf);
	    else
		printf ("Read symbols from \\"%s.\\"\\n", buf);
	    return;
	}
	if (strcasecmp (ext, ".obj") != 0) {
	    if (gui_mode)
		printf ("ERR {Only .obj files can be loaded.}\\n");
	    else
		printf ("Only .obj or .sym files can be loaded.\\n");
	    return;
	}
	#ifdef WASM
	unsigned char *debugext = strrchr (bufdebug, '.');
	strcpy(debugext, ".debug");
	#endif // WASM
    }
	#ifdef WASM
	if (read_obj_file (buf, bufdebug, &start, &end) == -1) {
	#else
	if (read_obj_file (buf, &start, &end) == -1) {
	#endif // WASM
	if (gui_mode)
	    printf ("ERR {Failed to load \\"%s.\\"}\\n", buf);
	else
	    printf ("Failed to load \\"%s.\\"\\n", buf);
	return;
    }
    /* Success: reload same file next time machine is reset. */
	#ifndef WASM
	if (start_file != NULL)
    	free (start_file);
    start_file = strdup (buf);
	#endif // !WASM
    
    strcpy (ext, ".sym");
    if (read_sym_file (buf))
        warn = 1;
    REG (R_PC) = start;

    // Clear the cache
    clear_cache ();

    /* GUI requires printing of new PC to reorient code display to line */
    if (gui_mode) {
	/* load new code into GUI display */
	disassemble (start, end);
	/* change focus in GUI */
	printf ("TOCODE\\n");
    	print_register (R_PC);
	if (warn)
	    printf ("ERR {WARNING: No symbols are available.}\\n");
    } else  {
	strcpy (ext, ".obj");
	printf ("Loaded \\"%s\\" and set PC to x%04X\\n", buf, start);
	if (warn)
	    printf ("WARNING: No symbols are available.\\n");
    }

	#ifdef WASM
	show_state_if_stop_visible();
	#endif // WASM

    /* Should not immediately start, even if we stopped simulator to
       load file.  We do need to update registers and dump delayed
       memory changes in that case, though.  Similarly, loading a
       file forces the simulator to forget completion of an executing
       "next" command. */
    if (interrupted_at_gui_request)
	gui_stop_and_dump ();
}


static void
cmd_finish (const char* args)
{
    no_args_allowed (args);
    flush_console_input ();
    finish_depth = 1;
    run_until_stopped ();
}


static void
cmd_help (const char* args) 
{
    printf ("file <file>           -- file load (also sets PC to start of "
    	    "file)\\n\\n");

    printf ("break ...             -- breakpoint management\\n\\n");

    printf ("continue              -- continue execution\\n");
    printf ("finish                -- execute to end of current subroutine\\n");
    printf ("next                  -- execute next instruction (full "
    	    "subroutine/trap)\\n");
    printf ("step                  -- execute one step (into "
    	    "subroutine/trap)\\n\\n");

    printf ("list ...              -- list instructions at the PC, an "
    	    "address, a label\\n");
    printf ("dump ...              -- dump memory at the PC, an address, "
    	    "a label\\n");
    printf ("translate <addr>      -- show the value of a label and print the "
    	    "contents\\n");
    printf ("printregs             -- print registers and current "
    	    "instruction\\n\\n");

    printf ("memory <addr> <val>   -- set the value held in a memory "
    	    "location\\n");
    printf ("register <reg> <val>  -- set a register to a value\\n\\n");

    printf ("rcontinue             -- continue execution in reverse\\n");
    printf ("rfinish               -- execute back to start of current subroutine\\n");
    printf ("rnext                 -- execute previous instruction (full "
    	    "subroutine/trap)\\n");
    printf ("rstep                 -- execute one step in reverse (into "
    	    "subroutine/trap)\\n\\n");

    printf ("execute <file name>   -- execute a script file\\n\\n");

    printf ("reset                 -- reset LC-3 and reload last file\\n\\n");

    printf ("quit                  -- quit the simulator\\n\\n");

    printf ("help                  -- print this help\\n\\n");

    printf ("All commands except quit can be abbreviated.\\n");
}


static int
parse_address (const char* addr)
{
    symbol_t* label;
    char* fmt;
    int value, negated;
    char trash[2];

    /* default matching order: symbol, hexadecimal */
    /* hexadecimal can optionally be preceded by x or X */
    /* decimal must be preceded by # */

    if (addr[0] == '-') {
	addr++;
	negated = 1;
    } else
	negated = 0;
    if ((label = find_symbol (addr, NULL)) != NULL)
        value = label->addr;
    else {
	if (*addr == '#')
	    fmt = "#%d%1s";
	else if (tolower (*addr) == 'x')
	    fmt = "x%x%1s";
	else
	    fmt = "%x%1s";
	if (sscanf (addr, fmt, &value, trash) != 1 || value > 0xFFFF ||
	    ((negated && value < 0) || (!negated && value < -0xFFFF)))
	    return -1;
    }
    if (negated)
        value = -value;
    if (value < 0)
	value += 0x10000;
    return value;
}


static int
parse_range (const char* args, int* startptr, int* endptr, 
             int last_end, int scale)
{
    char arg1[MAX_LABEL_LEN], arg2[MAX_LABEL_LEN], trash[2];
    int num_args, start, end;

    /* Split and count the arguments. */
    /* 80 == MAX_LABEL_LEN - 1 */
    num_args = sscanf (args, "%80s%80s%1s", arg1, arg2, trash);

    /* If we have no automatic scaling for the range, we
       need both the start and the end to be specified. */
    if (scale < 0 && num_args < 2)
	return -1;

    /* With no arguments, use automatic scaling around the PC. */
    if (num_args < 1) {
	start = (REG (R_PC) + 0x10000 - scale) & 0xFFFF;
	end = (REG (R_PC) + scale) & 0xFFFF;
	goto success;
    }

    /* If the first argument is "more," start from the last stopping
       point.   Note that "more" also requires automatic scaling. */
    if (last_end >= 0 && strcasecmp (arg1, "more") == 0) {
	start = last_end;
	end = (start + 2 * scale) & 0xFFFF;
	if (num_args > 1)
	    warn_too_many_args ();
	goto success;
    }

    /* Parse the starting address. */
    if ((start = parse_address (arg1)) == -1)
	return -1;

    /* Scale to find the ending address if necessary. */
    if (num_args < 2) {
	end = (start + 2 * scale) & 0xFFFF;
	goto success;
    }

    /* Parse the ending address. */
    if ((end = parse_address (arg2)) == -1)
        return -1;

    /* For ranges, add 1 to specified ending address for inclusion 
       in output. */
    if (scale >= 0)
	end = (end + 1) & 0xFFFF;

    /* Check for superfluous arguments. */
    if (num_args > 2)
	warn_too_many_args ();

    /* Store the results and return success. */ 
success:
    *startptr = start;
    *endptr = end;
    return 0;
}


static void 
cmd_list (const char* args)
{
    static int last_end = 0;
    int start, end;

    if (parse_range (args, &start, &end, last_end, 10) == 0) {
	disassemble (start, end);
	last_end = end;
	return;
    }

    printf ("list options include:\\n");
    printf ("  list               -- list instructions around PC\\n");
    printf ("  list <addr>        -- list instructions starting from an "
	    "address or label\\n");
    printf ("  list <addr> <addr> -- list a range of instructions\\n");
    printf ("  list more          -- continue previous listing (or press "
	    "<Enter>)\\n");
}


static void 
cmd_memory (const char* args)
{
    int addr, value;

    if (parse_range (args, &addr, &value, -1, -1) == 0) {
	// Cache the old value
	cur_mem_delta[0].valid = 1;
	cur_mem_delta[0].address = addr;
	cur_mem_delta[0].old_value = lc3_memory[addr];

	write_memory (addr, value);
	if (gui_mode) {
	    printf ("TRANS x%04X x%04X\\n", addr, value);
		#ifdef WASM
		disassemble_one (addr, 0);
		#else
		disassemble_one (addr);
		#endif // WASM
	} else
	    printf ("Wrote x%04X to address x%04X.\\n", value, addr);
		
	// Add this step to cache 
	cache_ (CACHE_TYPE_MEM);
    } else {
	if (gui_mode) {
	    /* Address is provided by the GUI, so only the value can
	       be bad in this case. */
	    printf ("ERR {No address or label corresponding to the "
		    "desired value exists.}\\n");
	} else
	    printf ("syntax: memory <addr> <value>\\n");
    }
}


static void 
cmd_option (const char* args)
{
    char opt[11], onoff[6], trash[2];
    int num_args, opt_len, oval;

    num_args = sscanf (args, "%10s%5s%1s", opt, onoff, trash);
    if (num_args >= 2) {
	opt_len = strlen (opt);
	if (strcasecmp (onoff, "on") == 0)
	    oval = 1;
	else if (strcasecmp (onoff, "off") == 0)
	    oval = 0;
	else
	    goto show_syntax;
        if (num_args > 2)
	    warn_too_many_args ();
        if (strncasecmp (opt, "flush", opt_len) == 0) {
	    flush_on_start = oval;
	    if (!gui_mode)
		printf ("Will %sflush the console input when starting.\\n",
			oval ? "" : "not ");
	    return;
	}
        if (strncasecmp (opt, "keep", opt_len) == 0) {
	    keep_input_on_stop = oval;
	    if (!gui_mode)
		printf ("Will %skeep remaining input when the LC-3 stops.\\n", 
			oval ? "" : "not ");
	    return;
	}
        if (strncasecmp (opt, "device", opt_len) == 0) {
	    rand_device = oval;
	    if (!gui_mode)
		printf ("Will %srandomize device interactions.\\n",
			oval ? "" : "not ");
	    return;
	}
	/* GUI-only option: Delay memory updates to GUI until LC-3 stops? */
        if (gui_mode && strncasecmp (opt, "delay", opt_len) == 0) {
	    /* Make sure that if the option is turned off while the GUI
	       thinks that the processor is running, state is dumped
	       immediately. */
	    if (delay_mem_update && oval == 0) 
		dump_delayed_mem_updates ();
	    delay_mem_update = oval;
	    return;
	}
	/* Use stdin for LC-3 console input while running script? */
        if (strncasecmp (opt, "stdin", opt_len) == 0) {
	    script_uses_stdin = oval;
	    if (!gui_mode)
		printf ("Will %suse stdin for LC-3 console input during "
			"script execution.\\n", oval ? "" : "not ");
	    if (script_depth > 0) {
	        if (!oval)
		    lc3in = sim_in;
		else if (!gui_mode)
		    lc3in = stdin;
		else
		    lc3in = lc3out;
	    }
	    return;
	}
    }

show_syntax:
    printf ("syntax: option <option> on|off\\n   options include:\\n");
    printf ("      device -- simulate random device (keyboard/display)"
    	    "timing\\n");
    printf ("      flush  -- flush console input each time LC-3 starts\\n");
    printf ("      keep   -- keep remaining input when the LC-3 stops\\n");
    printf ("      stdin  -- use stdin for LC-3 console input during script "
    	    "execution\\n");
    printf ("NOTE: all options are ON by default\\n");
}


static void 
cmd_next (const char* args)
{
    int next_pc = (REG (R_PC) + 1) & 0xFFFF;

    no_args_allowed (args);
    flush_console_input ();

    /* Note that we might hit a breakpoint immediately. */
    if (execute_instruction ()) {  
	if ((last_flags & FLG_SUBROUTINE) != 0) {
	    /* 
	       Mark system breakpoint.  This approach allows the GUI
	       to interrupt the simulator without the simulator 
	       forgetting about the completion of this command (i.e., 
	       next).  Nesting of such commands is not supported,
	       and should not be possible to issue with the GUI.
	    */
	   // Should be changed into finish
	    sys_bpt_addr = next_pc;
	    run_until_stopped ();
	    return;
	}
    }

    /* Dump memory and registers if necessary. */
    show_state_if_stop_visible ();
}


static void 
cmd_printregs (const char* args) 
{
    no_args_allowed (args);
    print_registers ();
}


static void 
cmd_quit (const char* args) 
{
    no_args_allowed (args);
    exit (0);
}


static void 
cmd_register (const char* args)
{
    char arg1[MAX_LABEL_LEN], arg2[MAX_LABEL_LEN], trash[2];
    int num_args, rnum, value, len;

    /* 80 == MAX_LABEL_LEN - 1 */
    num_args = sscanf (args, "%80s%80s%1s", arg1, arg2, trash);
    if (num_args < 2) {
	/* should never happen in GUI mode */
	printf ("syntax: register <reg> <value>\\n");
	return;
    }

    /* Determine which register is to be set. */
    for (rnum = 0; ; rnum++) {
	if (rnum == NUM_REGS + 1) {
	    /* No match (should never happen in GUI mode). */
	    puts ("Registers are R0...R7, PC, IR, PSR, and CC.");
	    return;
	}
	if (strcasecmp (rname[rnum], arg1) == 0)
	    break;
    }

    /* Condition codes are a special case. */
    if (rnum == NUM_REGS) {
	len = strlen (arg2);
	for (value = 0; value < 4; value++) {
	    if (strncasecmp (arg2, cc_val[value], len) == 0) {
		// Cache the current registers
		cur_reg_delta[0].valid = 1;
		cur_reg_delta[0].reg_num = R_PSR;
		cur_reg_delta[0].old_value = lc3_register[R_PSR];
		
		REG (R_PSR) &= ~0x0E00;
		REG (R_PSR) |= ((value + 1) << 9);

		cache_ (CACHE_TYPE_REG);

		if (gui_mode)
		    /* printing PSR prints both PSR and CC */
		    print_register (R_PSR);
		else
		    printf ("Set CC to %s.\\n", cc_val[value]);
		return;
	    }
	}
	if (gui_mode)
	    printf ("ERR {CC can only be set to NEGATIVE, ZERO, or "
		    "POSITIVE.}\\n");
	else
	    printf ("CC can only be set to NEGATIVE, ZERO, or "
		    "POSITIVE.\\n");
	return;
    } 

    /* Parse the value and set the register, or complain if it's a bad
       value. */
    if ((value = parse_address (arg2)) != -1) {
	// Cache the current registers
	cur_reg_delta[0].valid = 1;
	cur_reg_delta[0].reg_num = rnum;
	cur_reg_delta[0].old_value = lc3_register[rnum];

	REG (rnum) = value;

	if (rnum == R_PC) cache_(CACHE_TYPE_PC);
	else cache_ (CACHE_TYPE_REG);

	if (gui_mode)
	    print_register (rnum);
	else {
		#ifdef WASM
		char msg[30];
		sprintf(msg, "Set %s to x%04X.\\n", rname[rnum], value);
		EM_ASM_({
			reportSucc(\$0);
		},msg);
		#else
		printf ("Set %s to x%04X.\\n", rname[rnum], value);
		#endif // WASM
	}
    } else if (gui_mode)
	printf ("ERR {No address or label corresponding to the "
		"desired value exists.}\\n");
    else {
		#ifdef WASM
		EM_ASM_({reportErr(\$0);}, "No address or label corresponding to the desired value exists.");
		#else
		puts ("No address or label corresponding to the "
				"desired value exists.");
		#endif // WASM
	}
	
}


static void 
cmd_reset (const char* args)
{
    int addr;

    if (script_depth > 0) {
	/* Should never be executing a script in GUI mode, but check... */
	if (!gui_mode)
	    puts ("Cannot reset the LC-3 from within a script.");
    	else
	    puts ("ERR {Cannot reset the LC-3 from within a script.}");
    	return;
    }
    no_args_allowed (args);

    /* 
       If in GUI mode, we need to write over all memory with zeroes
       rather than just setting (so that disassembly info gets sent
       to GUI).
    */
    if (gui_mode) {
	interrupted_at_gui_request = 0;
        for (addr = 0; addr < 65536; addr++)
	    write_memory (addr, 0);
    	gui_stop_and_dump ();
    }

    /* various bits of state to reset */
    last_KBSR_read = 0;
    last_DSR_read = 0;
    have_mem_to_dump = 0;
    need_a_stop_notice = 0;
    sys_bpt_addr = -1;
    finish_depth = 0;

    init_machine ();

    /* change focus in GUI, and turn off delay cursor */
    if (gui_mode)
	printf ("TOCODE\\n");
}


static void 
cmd_step (const char* args)
{
    no_args_allowed (args);
    flush_console_input ();
    execute_instruction ();
    /* Dump memory and registers if necessary. */
    show_state_if_stop_visible ();
}


static void 
cmd_translate (const char* args)
{
    char arg1[81], trash[2];
    int num_args, value;

    /* 80 == MAX_LABEL_LEN - 1 */
    if ((num_args = sscanf (args, "%80s%1s", arg1, trash)) > 1)
    	warn_too_many_args ();

    if (num_args < 1) {
        puts ("syntax: translate <addr>");
	return;
    }

    /* Try to translate the value. */
    if ((value = parse_address (arg1)) == -1) {
	if (gui_mode)
	    printf ("ERR {No such address or label exists.}\\n");
	else
	    puts (BAD_ADDRESS);
    	return;
    }

    if (gui_mode)
	printf ("TRANS x%04X x%04X\\n", value, read_memory (value));
    else
	printf ("Address x%04X has value x%04x.\\n", value, 
		read_memory (value));
}


static void
gui_stop_and_dump ()
{
    /* Do not restart simulator automatically. */
    interrupted_at_gui_request = 0;

    /* Clear any breakpoint from an executing "next" command. */
    sys_bpt_addr = -1;

    /* Clear any "finish" command state. */
    finish_depth = 0;

    /* Tell the GUI about any changes to memory or registers. */
    dump_delayed_mem_updates ();
    print_registers ();
}


static void 
cmd_lc3_stop (const char* args)
{
    /* GUI only, so no need to warn about args. */
    /* Stop execution and dump state. */
    gui_stop_and_dump ();
}


static void 
flush_console_input ()
{
	#ifdef WASM
	return;
	#endif // WASM
    struct pollfd p;

    /* Check option and script level.  Flushing would consume 
       remainder of a script. */
    if (!flush_on_start || script_depth > 0)
        return;

    /* Read a character at a time... */
    p.fd = fileno (lc3in);
    p.events = POLLIN;
    while (poll (&p, 1, 0) == 1 && (p.revents & POLLIN) != 0)
	fgetc (lc3in);
}

static void 
cmd_r_step (const char* args)
{
    no_args_allowed (args);
    flush_console_input ();
    restore_step ();
	/* Dump memory and registers if necessary. */
	show_state_if_stop_visible ();
}

static void 
cmd_r_next (const char* args)
{
    no_args_allowed (args);
    flush_console_input ();

    int is_ret = (cache_last_flags () == FLG_RETURN);

    // Go back for one step. Stop if failed or hit a breakpoint
    if (restore_step () != 0){
	show_state_if_stop_visible ();
	return;
	}
	
    // If last instruction was RET, go back one step and r-finish. 
    if (is_ret) cmd_r_finish ("");
    else show_state_if_stop_visible ();
}

static void 
cmd_r_continue (const char* args)
{
    no_args_allowed (args);

#if 0
    // FIXME ... when interrupted, need to record direction of 
    // execution and do the right thing; at the moment, we don't
    // even stop execution in backwards direction when the GUI
    // needs attention...bad!
    if (interrupted_at_gui_request) 
	interrupted_at_gui_request = 0;
    else 
#endif
	flush_console_input ();
    
    while (restore_step () == 0) { /* empty */ }

    show_state_if_stop_visible ();
}

static void 
cmd_r_finish (const char* args)
{
    int end_depth = -1;
    int cur_depth = 0;

    no_args_allowed (args);
    flush_console_input ();

    // The depth will be updated after the inst with flags are restored, 
    // so use a variable to store
    inst_flag_t last_flag = cache_last_flags ();

    while (restore_step () == 0){
	// Just reversed a ret or reversed a call
	if (last_flag == FLG_RETURN) 
	    cur_depth++;
	if (last_flag == FLG_SUBROUTINE) {
	    // Check whether reach entry point
	    if (--cur_depth == end_depth) {
		break;
	    }
	}

	// Store for this step, useful in the next cycle
	last_flag = cache_last_flags ();
    }

    show_state_if_stop_visible ();
}

/**
 * The top level for caching current state 
 * Will determine whether to cache the whole state or just to cache deltas
 * @param	type: The type of current step. 0 for instruction, 1 for memory command, 2 for register command
 */ 
static void 
cache_ (cache_type_t type)
{
    cur_step++;
    cache_step (cur_step % MAX_CACHED_STEPS, type);
}

/**
 * Clear all the cache and set cur_step to 0
 */
static void 
clear_cache ()
{
    // The valid bits will also be set to 0
    bzero (cached_step, sizeof (cached_step));
    bzero (cur_reg_delta, sizeof (cur_reg_delta));
    bzero (cur_mem_delta, sizeof (cur_mem_delta));
    cur_step = 0;
}

/**
 * ONLY cache_ should call this function!
 * Add the delta for one step of instruction into cache 
 * @param	step: The step number to cache, should be cur_step%MAX_CACHED_STEPS
 * @return	0 for success, -1 for fail
 */
static int 
cache_step (int step, cache_type_t type)
{
    cached_step[step].flags.valid = 1;
    cached_step[step].flags.step = cur_step;
    cached_step[step].flags.pc = REG(R_PC);
    cached_step[step].flags.command = type;
    cached_step[step].flags.inst_flag = last_flags;

    // Check register delta, and clear the temp delta after cached 
    for (int i = 0; i < MAX_REG_CHANGED; i++){
	cached_step[step].reg_deltas[i] = cur_reg_delta[i];
	cur_reg_delta[i].valid = 0;
    }
    for (int i = 0; i < MAX_MEM_CHANGED; i++){
	cached_step[step].mem_deltas[i] = cur_mem_delta[i];
	cur_mem_delta[i].valid = 0;
    }
    
    return 0;
}

/**
 * Restore the delta by one step. After restored, the cached step will be cleared.
 * Will print out message if a a memory/register command is reset
 * @note	caller need to make sure correct step number is called. 
 * @param	step: The step whose generated delta will be restored 
 * @return 	-1 for fail, 0 for success and can continue, 1 for hitting a breakpoint
 */
static int 
restore_step ()
{
    int idx = (cur_step % MAX_CACHED_STEPS);

    if (cur_step < 1 || !cached_step[idx].flags.valid ||
        cached_step[idx].flags.step != cur_step) {
	if (gui_mode) {
	    printf ("ERR {Previous step is not tracked.  "
	            "Cannot go back anymore!}\\n");
	} else {
		#ifdef WASM
		EM_ASM_({reportWarn(\$0);}, "Previous step is not tracked.  Cannot go back anymore!\\n");
		#else
		printf ("Previous step is not tracked.  Cannot go back anymore!\\n");
		#endif // WASM
	    
	}
        return -1;
    }

    switch (cached_step[idx].flags.command) {
	case CACHE_TYPE_NONE:
	case CACHE_TYPE_INST: // instruction
	    break;
	case CACHE_TYPE_MEM: // memory
	    if (!gui_mode)
		printf ("Undid write x%04X to address x%04X.\\n",
			lc3_memory[cached_step[idx].mem_deltas[0].address],
			cached_step[idx].mem_deltas[0].address);
	    break;
	case CACHE_TYPE_PC: // no break on purpose
	case CACHE_TYPE_REG: // register
	    if (cached_step[idx].reg_deltas[0].reg_num == R_PSR) {
		if (!gui_mode)
		    printf ("Undid set CC to %s\\n", 
			    ccodes[(REG (R_PSR) >> 9) & 7]);
	    } else if (!gui_mode)
		printf ("Undid set %s to x%04X\\n",
			rname[cached_step[idx].reg_deltas[0].reg_num],
			lc3_register[cached_step[idx].reg_deltas[0].reg_num]);
	    break;
    }

    for (int i = 0; i < MAX_REG_CHANGED; i++) {
	reg_delta_t* cur = &cached_step[idx].reg_deltas[i];
	if (cur->valid) {
	    lc3_register[cur->reg_num] = cur->old_value;
	    if (gui_mode)
		print_register (cur->reg_num);
	}
    }

    for (int i = 0; i < MAX_MEM_CHANGED; i++){
	mem_delta_t* cur = &cached_step[idx].mem_deltas[i];
	if (cur->valid) {
	    have_mem_to_dump = 1;
		lc3_memory[cur->address] = cur->old_value;
	    if (gui_mode) {
        #ifdef WASM
        disassemble_one (cur->address, 0);
        #else
        disassemble_one (cur->address);
        #endif // WASM
	    }
	}
    }

    cached_step[idx].flags.valid = 0;
    cur_step--;

	if (lc3_breakpoints[REG (R_PC)] == BPT_USER && 
	    (cache_last_type () == CACHE_TYPE_INST ||  cache_last_type () == CACHE_TYPE_PC)) {
	    
		if (0 <= lc3_bpt_count[REG (R_PC)]) {
	    	lc3_bpt_count[REG (R_PC)]--;
		}
		if (0 > lc3_bpt_count[REG (R_PC)]) {
			if (!gui_mode)
			printf ("Reverse execution hit a breakpoint...\\n");
			return 1;
		}
	}
		
    
    return 0;
}

static inst_flag_t 
cache_last_flags ()
{
    int idx = (cur_step % MAX_CACHED_STEPS);

    if (cur_step < 1 || !cached_step[idx].flags.valid ||
        cached_step[idx].flags.step != cur_step) {
        return FLG_NONE;
    }
    return cached_step[idx].flags.inst_flag;
}

static cache_type_t 
cache_last_type ()
{
    int idx = (cur_step % MAX_CACHED_STEPS);

    if (cur_step < 1 || !cached_step[idx].flags.valid ||
        cached_step[idx].flags.step != cur_step) {
        return CACHE_TYPE_NONE;
    }
    return cached_step[idx].flags.command;
}
EOF
cat>lc3sim.h<<EOF
/*									tab:8
 *
 * lc3sim.h - the main header file for the LC-3 simulator
 *
 * "Copyright (c) 2003 by Steven S. Lumetta."
 *
 * Permission to use, copy, modify, and distribute this software and its
 * documentation for any purpose, without fee, and without written 
 * agreement is hereby granted, provided that the above copyright notice
 * and the following two paragraphs appear in all copies of this software,
 * that the files COPYING and NO_WARRANTY are included verbatim with
 * any distribution, and that the contents of the file README are included
 * verbatim as part of a file named README with any distribution.
 * 
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE TO ANY PARTY FOR DIRECT, 
 * INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT 
 * OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE AUTHOR 
 * HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 * THE AUTHOR SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT 
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR 
 * A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED HEREUNDER IS ON AN "AS IS" 
 * BASIS, AND THE AUTHOR NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, 
 * UPDATES, ENHANCEMENTS, OR MODIFICATIONS."
 *
 * Author:	    Steve Lumetta
 * Version:	    1
 * Creation Date:   18 October 2003
 * Filename:	    lc3sim.h
 * History:
 *	SSL	1	18 October 2003
 *		Copyright notices and Gnu Public License marker added.
 */

#ifndef LC3SIM_H
#define LC3SIM_H

/* field access macros; "i" is an instruction */

#define F_DR(i)    (((i) >> 9) & 0x7)
#define F_SR1(i)   (((i) >> 6) & 0x7)
#define F_SR2(i)   (((i) >> 0) & 0x7)
#define F_BaseR(i) F_SR1
#define F_SR(i)    F_DR    /* for stores */
#define F_CC(i)    ((i) & 0x0E00)
#define F_vec8(i)  ((i) & 0xFF)

#define F_imm5(i)  (((i) & 0x010) == 0 ? ((i) & 0x00F) : ((i) | ~0x00F))
#define F_imm6(i)  (((i) & 0x020) == 0 ? ((i) & 0x01F) : ((i) | ~0x01F))
#define F_imm9(i)  (((i) & 0x100) == 0 ? ((i) & 0x1FF) : ((i) | ~0x1FF))
#define F_imm11(i) (((i) & 0x400) == 0 ? ((i) & 0x3FF) : ((i) | ~0x3FF))


/* instruction fields for output formatting */

typedef enum field_t field_t;
enum field_t {
    FMT_R1    = 0x001, /* DR or SR                      */
    FMT_R2    = 0x002, /* SR1 or BaseR                  */
    FMT_R3    = 0x004, /* SR2                           */
    FMT_CC    = 0x008, /* condition codes               */
    FMT_IMM5  = 0x010, /* imm5                          */
    FMT_IMM6  = 0x020, /* imm6                          */
    FMT_VEC8  = 0x040, /* vec8                          */
    FMT_ASC8  = 0x080, /* 8-bit ASCII                   */
    FMT_IMM9  = 0x100, /* label (or address from imm9)  */
    FMT_IMM11 = 0x200, /* label (or address from imm11) */
    FMT_IMM16 = 0x400  /* full instruction in hex       */
};


/* instruction formats for output */

typedef enum format_t format_t;
enum format_t {
    FMT_      = 0,
    FMT_RRR   = (FMT_R1 | FMT_R2 | FMT_R3),
    FMT_RRI   = (FMT_R1 | FMT_R2 | FMT_IMM5),
    FMT_CL    = (FMT_CC | FMT_IMM9),
    FMT_R     = FMT_R2,
    FMT_L     = FMT_IMM11,
    FMT_RL    = (FMT_R1 | FMT_IMM9),
    FMT_RRI6  = (FMT_R1 | FMT_R2 | FMT_IMM6),
    FMT_RR    = (FMT_R1 | FMT_R2),
    FMT_V     = FMT_VEC8,
    FMT_A     = FMT_ASC8,
    FMT_16    = FMT_IMM16
};


/* instruction flags */

typedef enum inst_flag_t inst_flag_t;
enum inst_flag_t {
    FLG_NONE       = 0,
    FLG_SUBROUTINE = 1,
    FLG_RETURN     = 2
};


/* LC-3 registers */

typedef enum reg_num_t reg_num_t;
enum reg_num_t {
	R_R0 = 0, R_R1, R_R2, R_R3, R_R4, R_R5, R_R6, R_R7,
	R_PC, R_IR, R_PSR,
	NUM_REGS
};


extern int read_memory (int addr);
extern void write_memory (int addr, int value);


#endif /* LC3SIM_H */

EOF
cat>Makefile.def<<EOF
###############################################################################
# Makefile (and Makefile.def) -- Makefile for the LC-3 tools
#
#  "Copyright (c) 2003-2020 by Steven S. Lumetta."
# 
#  Permission to use, copy, modify, and distribute this software and its
#  documentation for any purpose, without fee, and without written 
#  agreement is hereby granted, provided that the above copyright notice
#  and the following two paragraphs appear in all copies of this software,
#  that the files COPYING and NO_WARRANTY are included verbatim with
#  any distribution, and that the contents of the file README are included
#  verbatim as part of a file named README with any distribution.
#  
#  IN NO EVENT SHALL THE AUTHOR BE LIABLE TO ANY PARTY FOR DIRECT, 
#  INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT 
#  OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE AUTHOR 
#  HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#  
#  THE AUTHOR SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT 
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR 
#  A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED HEREUNDER IS ON AN "AS IS" 
#  BASIS, AND THE AUTHOR NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, 
#  UPDATES, ENHANCEMENTS, OR MODIFICATIONS."
#
#  Author:		Steve Lumetta
#  Version:		3
#  Creation Date:	18 October 2003
#  Filename:		Makefile[.def]
#  History:		
# 	SSL	1	18 October 2003
# 		Copyright notices and Gnu Public License marker added.
# 	SSL	2	31 October 2003
# 		Added lc3convert tool into distribution.
# 	SSL	3	9 October 2020
# 		Added -DYY_NO_INPUT=1 to CFLAGS.
#
###############################################################################

# These path names are automatically set by configure.
GCC  = __GCC__
FLEX = __FLEX__
EXE  = __EXE__
OS_SIM_LIBS = __OS_SIM_LIBS__
RM   = __RM__
CP   = __CP__
MKDIR = __MKDIR__
CHMOD = __CHMOD__
SED = __SED__
WISH = __WISH__
RLIPATH = __RLIPATH__
RLLPATH = __RLLPATH__
USE_READLINE = __USE_READLINE__
INSTALL_DIR = __INSTALL_DIR__
CODE_FONT = __CODE_FONT__
BUTTON_FONT = __BUTTON_FONT__
CONSOLE_FONT = __CONSOLE_FONT__
# End of configuration block.

CFLAGS  = -g -Wall -DYY_NO_INPUT=1
LDFLAGS = -g
LC3AS   = ./lc3as

ALL: dist_lc3as dist_lc3convert dist_lc3sim dist_lc3sim-tk

clean: dist_lc3as_clean dist_lc3convert_clean dist_lc3sim_clean \\
	dist_lc3sim-tk_clean

clear: dist_lc3as_clear dist_lc3convert_clear dist_lc3sim_clear \\
	dist_lc3sim-tk_clear

distclean: clean clear
	\${RM} -f Makefile

install: ALL
	\${MKDIR} -p \${INSTALL_DIR}
	-\${CP} -f lc3as\${EXE} lc3convert\${EXE} lc3sim\${EXE} lc3os.obj \\
		lc3os.sym lc3sim-tk COPYING NO_WARRANTY README \${INSTALL_DIR}
	\${CHMOD} 555 \${INSTALL_DIR}/lc3as\${EXE} \\
		\${INSTALL_DIR}/lc3convert\${EXE} \${INSTALL_DIR}/lc3sim\${EXE} \\
		\${INSTALL_DIR}/lc3sim-tk
	\${CHMOD} 444 \${INSTALL_DIR}/lc3os.obj \${INSTALL_DIR}/lc3os.sym \\
		\${INSTALL_DIR}/COPYING \${INSTALL_DIR}/NO_WARRANTY      \\
		\${INSTALL_DIR}/README

%.o: %.c
	\${GCC} -c \${CFLAGS} -o \$@ \$<

#
# Makefile fragment for lc3as
#

dist_lc3as: lc3as\${EXE}

lc3as\${EXE}: lex.lc3.o symbol.o
	\${GCC} \${LDFLAGS} -o lc3as\${EXE} lex.lc3.o symbol.o

lex.lc3.c: lc3.f
	\${FLEX} -i -Plc3 lc3.f

dist_lc3as_clean::
	\${RM} -f lex.lc3.c *.o *~

dist_lc3as_clear: dist_lc3as_clean
	\${RM} -f lc3as\${EXE}

#
# Makefile fragment for lc3convert
#

dist_lc3convert: lc3convert\${EXE}

lc3convert\${EXE}: lex.lc3convert.o
	\${GCC} \${LDFLAGS} -o lc3convert\${EXE} lex.lc3convert.o

lex.lc3convert.c: lc3convert.f
	\${FLEX} -i -Plc3convert lc3convert.f

dist_lc3convert_clean::
	\${RM} -f lex.lc3convert.c *.o *~

dist_lc3convert_clear: dist_lc3convert_clean
	\${RM} -f lc3convert\${EXE}

#
# Makefile fragment for lc3sim
#

dist_lc3sim: lc3sim\${EXE} lc3os.obj lc3os.sym

lc3sim\${EXE}: lc3sim.o sim_symbol.o
	\${GCC} \${LDFLAGS} \${RLIPATH} -o lc3sim\${EXE} \\
		lc3sim.o sim_symbol.o \${RLLPATH} \${OS_SIM_LIBS}

lc3os.obj: \${LC3AS} lc3os.asm
	\${LC3AS} lc3os

lc3os.sym: \${LC3AS} lc3os.asm
	\${LC3AS} lc3os

lc3sim.o: lc3sim.c lc3.def lc3sim.h symbol.h
	\${GCC} -c \${CFLAGS} \${USE_READLINE} -DINSTALL_DIR="\\"\${INSTALL_DIR}\\"" -DMAP_LOCATION_TO_SYMBOL -o lc3sim.o lc3sim.c

sim_symbol.o: symbol.c symbol.h
	\${GCC} -c \${CFLAGS} -DMAP_LOCATION_TO_SYMBOL -o sim_symbol.o symbol.c

dist_lc3sim_clean::
	\${RM} -f *.o *~

dist_lc3sim_clear: dist_lc3sim_clean
	\${RM} -f lc3sim\${EXE} lc3os.obj lc3os.sym

#
# Makefile fragment for lc3sim-tk
#

dist_lc3sim-tk: lc3sim-tk

lc3sim-tk: lc3sim-tk.def
	\${SED} -e 's @@WISH@@ \${WISH} g' \\
		-e 's*@@LC3_SIM@@*"\${INSTALL_DIR}/lc3sim"*g' \\
		-e 's!@@CODE_FONT@@!\${CODE_FONT}!g' \\
		-e 's!@@BUTTON_FONT@@!\${BUTTON_FONT}!g' \\
		-e 's!@@CONSOLE_FONT@@!\${CONSOLE_FONT}!g' \\
		lc3sim-tk.def > lc3sim-tk
	\${CHMOD} u+x lc3sim-tk

dist_lc3sim-tk_clean::

dist_lc3sim-tk_clear::
	\${RM} -f lc3sim-tk

EOF
cat>symbol.c<<EOF
/*									tab:8
 *
 * symbol.c - symbol table functions for the LC-3 assembler and simulator
 *
 * "Copyright (c) 2003-2020 by Steven S. Lumetta."
 *
 * Permission to use, copy, modify, and distribute this software and its
 * documentation for any purpose, without fee, and without written 
 * agreement is hereby granted, provided that the above copyright notice
 * and the following two paragraphs appear in all copies of this software,
 * that the files COPYING and NO_WARRANTY are included verbatim with
 * any distribution, and that the contents of the file README are included
 * verbatim as part of a file named README with any distribution.
 * 
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE TO ANY PARTY FOR DIRECT, 
 * INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT 
 * OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE AUTHOR 
 * HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 * THE AUTHOR SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT 
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR 
 * A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED HEREUNDER IS ON AN "AS IS" 
 * BASIS, AND THE AUTHOR NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, 
 * UPDATES, ENHANCEMENTS, OR MODIFICATIONS."
 *
 * Author:	    Steve Lumetta
 * Version:	    1
 * Creation Date:   18 October 2003
 * Filename:	    symbol.c
 * History:
 *	SSL	1	18 October 2003
 *		Copyright notices and Gnu Public License marker added.
 *	SSL	2	9 October 2020
 *		Eliminated warning for empty loop body.
 */

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

#include "symbol.h"

symbol_t* lc3_sym_hash[SYMBOL_HASH];
#ifdef MAP_LOCATION_TO_SYMBOL
symbol_t* lc3_sym_names[65536];
#endif

int
symbol_hash (const char* symbol)
{
    int h = 1;

    while (*symbol != 0)
	h = (h * tolower (*symbol++)) % SYMBOL_HASH;

    return h;
}

symbol_t*
find_symbol (const char* symbol, int* hptr)
{
    int h = symbol_hash (symbol);
    symbol_t* sym;

    if (hptr != NULL)
        *hptr = h;
    for (sym = lc3_sym_hash[h]; sym != NULL; sym = sym->next_with_hash)
    	if (strcasecmp (symbol, sym->name) == 0)
	    return sym;
    return NULL;
}

int
add_symbol (const char* symbol, int addr, int dup_ok)
{
    int h;
    symbol_t* sym;

    if ((sym = find_symbol (symbol, &h)) == NULL) {
	sym = (symbol_t*)malloc (sizeof (symbol_t)); 
	sym->name = strdup (symbol);
	sym->next_with_hash = lc3_sym_hash[h];
	lc3_sym_hash[h] = sym;
#ifdef MAP_LOCATION_TO_SYMBOL
	sym->next_at_loc = lc3_sym_names[addr];
	lc3_sym_names[addr] = sym;
#endif
    } else if (!dup_ok) 
        return -1;
    sym->addr = addr;
    return 0;
}


#ifdef MAP_LOCATION_TO_SYMBOL
void
remove_symbol_at_addr (int addr)
{
    symbol_t* s;
    symbol_t** find;
    int h;

    while ((s = lc3_sym_names[addr]) != NULL) {
        h = symbol_hash (s->name);
	for (find = &lc3_sym_hash[h]; *find != s; 
	     find = &(*find)->next_with_hash) { }
        *find = s->next_with_hash;
	lc3_sym_names[addr] = s->next_at_loc;
	free (s->name);
	free (s);
    }
}
#endif

EOF
cat>symbol.h<<EOF
/*									tab:8
 *
 * symbol.h - symbol table interface for the LC-3 assembler and simulator
 *
 * "Copyright (c) 2003 by Steven S. Lumetta."
 *
 * Permission to use, copy, modify, and distribute this software and its
 * documentation for any purpose, without fee, and without written 
 * agreement is hereby granted, provided that the above copyright notice
 * and the following two paragraphs appear in all copies of this software,
 * that the files COPYING and NO_WARRANTY are included verbatim with
 * any distribution, and that the contents of the file README are included
 * verbatim as part of a file named README with any distribution.
 * 
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE TO ANY PARTY FOR DIRECT, 
 * INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT 
 * OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE AUTHOR 
 * HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 * THE AUTHOR SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT 
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR 
 * A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED HEREUNDER IS ON AN "AS IS" 
 * BASIS, AND THE AUTHOR NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, 
 * UPDATES, ENHANCEMENTS, OR MODIFICATIONS."
 *
 * Author:	    Steve Lumetta
 * Version:	    1
 * Creation Date:   18 October 2003
 * Filename:	    symbol.h
 * History:
 *	SSL	1	18 October 2003
 *		Copyright notices and Gnu Public License marker added.
 */

#ifndef SYMBOL_H
#define SYMBOL_H

typedef struct symbol_t symbol_t;
struct symbol_t {
    char* name;
    int addr;
    symbol_t* next_with_hash;
#ifdef MAP_LOCATION_TO_SYMBOL
    symbol_t* next_at_loc;
#endif
};

#define SYMBOL_HASH 997

extern symbol_t* lc3_sym_names[65536];
extern symbol_t* lc3_sym_hash[SYMBOL_HASH];

int symbol_hash (const char* symbol);
int add_symbol (const char* symbol, int addr, int dup_ok);
symbol_t* find_symbol (const char* symbol, int* hptr);
#ifdef MAP_LOCATION_TO_SYMBOL
void remove_symbol_at_addr (int addr);
#endif

#endif /* SYMBOL_H */

EOF

echo -e "\nConfiguring environment..."
chmod +x ./configure
./configure --installdir /usr/local/bin > /dev/null

echo -e "\nCompiling files..."
make CFLAGS=-g LDFLAGS=-lreadline USE_READLINE=-DUSE_READLINE > /dev/null

echo -e "\nBuilding the module..."
sudo make install > /dev/null

rm -r ../lc3tools
echo -e "\nComplete.\n"
