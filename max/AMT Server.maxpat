{
	"patcher" : 	{
		"fileversion" : 1,
		"appversion" : 		{
			"major" : 8,
			"minor" : 6,
			"revision" : 5,
			"architecture" : "x64",
			"modernui" : 1
		}
,
		"classnamespace" : "box",
		"rect" : [ 886.0, 189.0, 1000.0, 780.0 ],
		"bglocked" : 0,
		"openinpresentation" : 0,
		"default_fontsize" : 12.0,
		"default_fontface" : 0,
		"default_fontname" : "Arial",
		"gridonopen" : 1,
		"gridsize" : [ 15.0, 15.0 ],
		"gridsnaponopen" : 1,
		"objectsnaponopen" : 1,
		"statusbarvisible" : 2,
		"toolbarvisible" : 1,
		"lefttoolbarpinned" : 0,
		"toptoolbarpinned" : 0,
		"righttoolbarpinned" : 0,
		"bottomtoolbarpinned" : 0,
		"toolbars_unpinned_last_save" : 0,
		"tallnewobj" : 0,
		"boxanimatetime" : 200,
		"enablehscroll" : 1,
		"enablevscroll" : 1,
		"devicewidth" : 0.0,
		"description" : "",
		"digest" : "",
		"tags" : "",
		"style" : "",
		"subpatcher_template" : "",
		"assistshowspatchername" : 0,
		"boxes" : [ 			{
				"box" : 				{
					"id" : "obj-1",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 0,
					"patching_rect" : [ 212.800003170967102, 193.600002884864807, 49.0, 22.0 ],
					"text" : "noteout"
				}

			}
, 			{
				"box" : 				{
					"fontface" : 1,
					"fontsize" : 12.0,
					"id" : "lbl-send",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 20.0, 14.0, 200.0, 20.0 ],
					"text" : "■  MIDI  →  GPU SERVER"
				}

			}
, 			{
				"box" : 				{
					"fontsize" : 10.0,
					"id" : "lbl-note-ip",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 20.0, 194.0, 175.0, 18.0 ],
					"text" : "▲ change IP/port here"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-notein",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 3,
					"outlettype" : [ "int", "int", "int" ],
					"patching_rect" : [ 20.0, 44.0, 55.0, 22.0 ],
					"text" : "notein"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-pack",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 20.0, 88.0, 60.0, 22.0 ],
					"text" : "pack i i"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-prepend-note",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 20.0, 132.0, 105.0, 22.0 ],
					"text" : "prepend /note"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-udpsend",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 20.0, 172.0, 158.0, 22.0 ],
					"text" : "udpsend 192.168.1.10 9000"
				}

			}
, 			{
				"box" : 				{
					"fontface" : 1,
					"fontsize" : 12.0,
					"id" : "lbl-recv",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 310.0, 14.0, 210.0, 20.0 ],
					"text" : "■  GPU SERVER  →  MIDI"
				}

			}
, 			{
				"box" : 				{
					"fontsize" : 10.0,
					"id" : "lbl-ch2",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 310.0, 230.0, 250.0, 18.0 ],
					"text" : "output on MIDI ch 2+  (ch 1 = human)"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-udprecv",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 310.0, 44.0, 120.0, 22.0 ],
					"text" : "udpreceive 9001"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-route",
					"maxclass" : "newobj",
					"numinlets" : 4,
					"numoutlets" : 4,
					"outlettype" : [ "", "", "", "" ],
					"patching_rect" : [ 310.0, 88.0, 310.0, 22.0 ],
					"text" : "route /gen/noteon /gen/noteoff /gen/status"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-unpack-on",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 3,
					"outlettype" : [ "int", "int", "int" ],
					"patching_rect" : [ 310.0, 132.0, 105.0, 22.0 ],
					"text" : "unpack i i i"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-noteout-on",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 0,
					"patching_rect" : [ 310.0, 200.0, 55.0, 22.0 ],
					"text" : "noteout"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-unpack-off",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "int", "int" ],
					"patching_rect" : [ 470.0, 132.0, 85.0, 22.0 ],
					"text" : "unpack i i"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-int-off-vel",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"patching_rect" : [ 520.0, 162.0, 25.0, 22.0 ],
					"text" : "i"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-noteout-off",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 0,
					"patching_rect" : [ 470.0, 200.0, 55.0, 22.0 ],
					"text" : "noteout"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-print-status",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 630.0, 132.0, 125.0, 22.0 ],
					"text" : "print gen_status"
				}

			}
, 			{
				"box" : 				{
					"fontface" : 1,
					"fontsize" : 12.0,
					"id" : "lbl-session",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 20.0, 268.0, 130.0, 20.0 ],
					"text" : "■  SESSION"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-toggle",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 20.0, 298.0, 30.0, 30.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "lbl-toggle",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 62.0, 304.0, 100.0, 20.0 ],
					"text" : "START / STOP"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-sel",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 3,
					"outlettype" : [ "bang", "bang", "" ],
					"patching_rect" : [ 20.0, 348.0, 55.0, 22.0 ],
					"text" : "sel 0 1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-msg-stop",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 20.0, 390.0, 110.0, 22.0 ],
					"text" : "/control/stop"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-msg-start",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 145.0, 390.0, 115.0, 22.0 ],
					"text" : "/control/start"
				}

			}
, 			{
				"box" : 				{
					"fontface" : 1,
					"fontsize" : 12.0,
					"id" : "lbl-params",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 20.0, 435.0, 240.0, 20.0 ],
					"text" : "■  GENERATION PARAMETERS"
				}

			}
, 			{
				"box" : 				{
					"id" : "lbl-win",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 20.0, 465.0, 95.0, 20.0 ],
					"text" : "window (sec)"
				}

			}
, 			{
				"box" : 				{
					"format" : 6,
					"id" : "obj-flonum-win",
					"maxclass" : "flonum",
					"maximum" : 20.0,
					"minimum" : 2.0,
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 120.0, 462.0, 55.0, 22.0 ],
					"varname" : "6."
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-prepend-win",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 20.0, 498.0, 215.0, 22.0 ],
					"text" : "prepend /control/window_size"
				}

			}
, 			{
				"box" : 				{
					"id" : "lbl-topp",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 270.0, 465.0, 48.0, 20.0 ],
					"text" : "top_p"
				}

			}
, 			{
				"box" : 				{
					"format" : 6,
					"id" : "obj-flonum-topp",
					"maxclass" : "flonum",
					"maximum" : 1.0,
					"minimum" : 0.5,
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 320.0, 462.0, 55.0, 22.0 ],
					"varname" : "0.95"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-prepend-topp",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 270.0, 498.0, 175.0, 22.0 ],
					"text" : "prepend /control/top_p"
				}

			}
, 			{
				"box" : 				{
					"id" : "lbl-temp",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 490.0, 465.0, 85.0, 20.0 ],
					"text" : "temperature"
				}

			}
, 			{
				"box" : 				{
					"format" : 6,
					"id" : "obj-flonum-temp",
					"maxclass" : "flonum",
					"maximum" : 2.0,
					"minimum" : 0.5,
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 578.0, 462.0, 55.0, 22.0 ],
					"varname" : "1."
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-prepend-temp",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 490.0, 498.0, 215.0, 22.0 ],
					"text" : "prepend /control/temperature"
				}

			}
 ],
		"lines" : [ 			{
				"patchline" : 				{
					"destination" : [ "obj-prepend-temp", 0 ],
					"source" : [ "obj-flonum-temp", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-prepend-topp", 0 ],
					"source" : [ "obj-flonum-topp", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-prepend-win", 0 ],
					"source" : [ "obj-flonum-win", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-noteout-off", 1 ],
					"source" : [ "obj-int-off-vel", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-udpsend", 0 ],
					"source" : [ "obj-msg-start", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-udpsend", 0 ],
					"source" : [ "obj-msg-stop", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-1", 1 ],
					"order" : 0,
					"source" : [ "obj-notein", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-1", 0 ],
					"order" : 0,
					"source" : [ "obj-notein", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-pack", 1 ],
					"order" : 1,
					"source" : [ "obj-notein", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-pack", 0 ],
					"order" : 1,
					"source" : [ "obj-notein", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-prepend-note", 0 ],
					"source" : [ "obj-pack", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-udpsend", 0 ],
					"source" : [ "obj-prepend-note", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-udpsend", 0 ],
					"source" : [ "obj-prepend-temp", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-udpsend", 0 ],
					"source" : [ "obj-prepend-topp", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-udpsend", 0 ],
					"source" : [ "obj-prepend-win", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-print-status", 0 ],
					"source" : [ "obj-route", 2 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-unpack-off", 0 ],
					"source" : [ "obj-route", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-unpack-on", 0 ],
					"source" : [ "obj-route", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-msg-start", 0 ],
					"source" : [ "obj-sel", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-msg-stop", 0 ],
					"source" : [ "obj-sel", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-sel", 0 ],
					"source" : [ "obj-toggle", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-route", 0 ],
					"source" : [ "obj-udprecv", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-noteout-off", 2 ],
					"source" : [ "obj-unpack-off", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-noteout-off", 0 ],
					"source" : [ "obj-unpack-off", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-noteout-on", 2 ],
					"source" : [ "obj-unpack-on", 2 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-noteout-on", 1 ],
					"source" : [ "obj-unpack-on", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-noteout-on", 0 ],
					"source" : [ "obj-unpack-on", 0 ]
				}

			}
 ],
		"dependency_cache" : [  ],
		"autosave" : 0
	}

}
