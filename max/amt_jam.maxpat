{
	"patcher" : 	{
		"fileversion" : 1,
		"appversion" : 		{
			"major" : 8,
			"minor" : 6,
			"bugfix" : 4,
			"type" : "release",
			"moddate" : 0,
			"target" : ""
		}
,
		"classnamespace" : "box",
		"rect" : [ 59.0, 82.0, 740.0, 600.0 ],
		"bglocked" : 0,
		"openinpresentation" : 0,
		"default_fontsize" : 12.0,
		"default_fontface" : 0,
		"default_fontname" : "Arial",
		"gridonopen" : 1,
		"gridsize" : [ 15.0, 15.0 ],
		"gridsnaponopen" : 1,
		"objectsnapoffset" : [ 0.0, 0.0 ],
		"fontpanelonopen" : 0,
		"mousescrollmode" : 0,
		"prefcountmaxobj" : 0,
		"boxes" : [
			{
				"box" : 				{
					"id" : "lbl-send",
					"maxclass" : "comment",
					"text" : "■  MIDI  →  GPU SERVER",
					"numinlets" : 1,
					"numoutlets" : 0,
					"fontface" : 1,
					"fontsize" : 12.0,
					"patching_rect" : [ 20.0, 14.0, 200.0, 20.0 ]
				}
			},
			{
				"box" : 				{
					"id" : "lbl-note-ip",
					"maxclass" : "comment",
					"text" : "▲ change IP/port here",
					"numinlets" : 1,
					"numoutlets" : 0,
					"fontsize" : 10.0,
					"patching_rect" : [ 20.0, 194.0, 175.0, 18.0 ]
				}
			},
			{
				"box" : 				{
					"id" : "obj-notein",
					"maxclass" : "newobj",
					"text" : "notein",
					"numinlets" : 0,
					"numoutlets" : 3,
					"outlettype" : [ "int", "int", "int" ],
					"patching_rect" : [ 20.0, 44.0, 55.0, 22.0 ]
				}
			},
			{
				"box" : 				{
					"id" : "obj-pack",
					"maxclass" : "newobj",
					"text" : "pack i i",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "list" ],
					"patching_rect" : [ 20.0, 88.0, 60.0, 22.0 ]
				}
			},
			{
				"box" : 				{
					"id" : "obj-prepend-note",
					"maxclass" : "newobj",
					"text" : "prepend /note",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 20.0, 132.0, 105.0, 22.0 ]
				}
			},
			{
				"box" : 				{
					"id" : "obj-udpsend",
					"maxclass" : "newobj",
					"text" : "udpsend SERVER_IP 9000",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 20.0, 172.0, 178.0, 22.0 ]
				}
			},
			{
				"box" : 				{
					"id" : "lbl-recv",
					"maxclass" : "comment",
					"text" : "■  GPU SERVER  →  MIDI",
					"numinlets" : 1,
					"numoutlets" : 0,
					"fontface" : 1,
					"fontsize" : 12.0,
					"patching_rect" : [ 310.0, 14.0, 210.0, 20.0 ]
				}
			},
			{
				"box" : 				{
					"id" : "lbl-ch2",
					"maxclass" : "comment",
					"text" : "output on MIDI ch 2+  (ch 1 = human)",
					"numinlets" : 1,
					"numoutlets" : 0,
					"fontsize" : 10.0,
					"patching_rect" : [ 310.0, 230.0, 250.0, 18.0 ]
				}
			},
			{
				"box" : 				{
					"id" : "obj-udprecv",
					"maxclass" : "newobj",
					"text" : "udpreceive 9001",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 310.0, 44.0, 120.0, 22.0 ]
				}
			},
			{
				"box" : 				{
					"id" : "obj-route",
					"maxclass" : "newobj",
					"text" : "route /gen/noteon /gen/noteoff /gen/status",
					"numinlets" : 1,
					"numoutlets" : 4,
					"outlettype" : [ "", "", "", "" ],
					"patching_rect" : [ 310.0, 88.0, 310.0, 22.0 ]
				}
			},
			{
				"box" : 				{
					"id" : "obj-unpack-on",
					"maxclass" : "newobj",
					"text" : "unpack i i i",
					"numinlets" : 1,
					"numoutlets" : 3,
					"outlettype" : [ "int", "int", "int" ],
					"patching_rect" : [ 310.0, 132.0, 105.0, 22.0 ]
				}
			},
			{
				"box" : 				{
					"id" : "obj-noteout-on",
					"maxclass" : "newobj",
					"text" : "noteout",
					"numinlets" : 3,
					"numoutlets" : 0,
					"patching_rect" : [ 310.0, 200.0, 55.0, 22.0 ]
				}
			},
			{
				"box" : 				{
					"id" : "obj-unpack-off",
					"maxclass" : "newobj",
					"text" : "unpack i i",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "int", "int" ],
					"patching_rect" : [ 470.0, 132.0, 85.0, 22.0 ]
				}
			},
			{
				"box" : 				{
					"id" : "obj-int-off-vel",
					"maxclass" : "newobj",
					"text" : "i",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"patching_rect" : [ 520.0, 162.0, 25.0, 22.0 ]
				}
			},
			{
				"box" : 				{
					"id" : "obj-noteout-off",
					"maxclass" : "newobj",
					"text" : "noteout",
					"numinlets" : 3,
					"numoutlets" : 0,
					"patching_rect" : [ 470.0, 200.0, 55.0, 22.0 ]
				}
			},
			{
				"box" : 				{
					"id" : "obj-print-status",
					"maxclass" : "newobj",
					"text" : "print gen_status",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 630.0, 132.0, 125.0, 22.0 ]
				}
			},
			{
				"box" : 				{
					"id" : "lbl-session",
					"maxclass" : "comment",
					"text" : "■  SESSION",
					"numinlets" : 1,
					"numoutlets" : 0,
					"fontface" : 1,
					"fontsize" : 12.0,
					"patching_rect" : [ 20.0, 268.0, 130.0, 20.0 ]
				}
			},
			{
				"box" : 				{
					"id" : "obj-toggle",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"patching_rect" : [ 20.0, 298.0, 30.0, 30.0 ]
				}
			},
			{
				"box" : 				{
					"id" : "lbl-toggle",
					"maxclass" : "comment",
					"text" : "START / STOP",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 62.0, 304.0, 100.0, 20.0 ]
				}
			},
			{
				"box" : 				{
					"id" : "obj-sel",
					"maxclass" : "newobj",
					"text" : "sel 0 1",
					"numinlets" : 1,
					"numoutlets" : 3,
					"outlettype" : [ "bang", "bang", "" ],
					"patching_rect" : [ 20.0, 348.0, 55.0, 22.0 ]
				}
			},
			{
				"box" : 				{
					"id" : "obj-msg-stop",
					"maxclass" : "message",
					"text" : "/control/stop",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 20.0, 390.0, 110.0, 22.0 ]
				}
			},
			{
				"box" : 				{
					"id" : "obj-msg-start",
					"maxclass" : "message",
					"text" : "/control/start",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 145.0, 390.0, 115.0, 22.0 ]
				}
			},
			{
				"box" : 				{
					"id" : "lbl-params",
					"maxclass" : "comment",
					"text" : "■  GENERATION PARAMETERS",
					"numinlets" : 1,
					"numoutlets" : 0,
					"fontface" : 1,
					"fontsize" : 12.0,
					"patching_rect" : [ 20.0, 435.0, 240.0, 20.0 ]
				}
			},
			{
				"box" : 				{
					"id" : "lbl-win",
					"maxclass" : "comment",
					"text" : "window (sec)",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 20.0, 465.0, 95.0, 20.0 ]
				}
			},
			{
				"box" : 				{
					"id" : "obj-flonum-win",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "float", "bang" ],
					"minimum" : 2.0,
					"maximum" : 20.0,
					"patching_rect" : [ 120.0, 462.0, 55.0, 22.0 ],
					"varname" : "6."
				}
			},
			{
				"box" : 				{
					"id" : "obj-prepend-win",
					"maxclass" : "newobj",
					"text" : "prepend /control/window_size",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 20.0, 498.0, 215.0, 22.0 ]
				}
			},
			{
				"box" : 				{
					"id" : "lbl-topp",
					"maxclass" : "comment",
					"text" : "top_p",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 270.0, 465.0, 48.0, 20.0 ]
				}
			},
			{
				"box" : 				{
					"id" : "obj-flonum-topp",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "float", "bang" ],
					"minimum" : 0.5,
					"maximum" : 1.0,
					"patching_rect" : [ 320.0, 462.0, 55.0, 22.0 ],
					"varname" : "0.95"
				}
			},
			{
				"box" : 				{
					"id" : "obj-prepend-topp",
					"maxclass" : "newobj",
					"text" : "prepend /control/top_p",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 270.0, 498.0, 175.0, 22.0 ]
				}
			},
			{
				"box" : 				{
					"id" : "lbl-temp",
					"maxclass" : "comment",
					"text" : "temperature",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 490.0, 465.0, 85.0, 20.0 ]
				}
			},
			{
				"box" : 				{
					"id" : "obj-flonum-temp",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "float", "bang" ],
					"minimum" : 0.5,
					"maximum" : 2.0,
					"patching_rect" : [ 578.0, 462.0, 55.0, 22.0 ],
					"varname" : "1."
				}
			},
			{
				"box" : 				{
					"id" : "obj-prepend-temp",
					"maxclass" : "newobj",
					"text" : "prepend /control/temperature",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 490.0, 498.0, 215.0, 22.0 ]
				}
			}
		],
		"lines" : [
			{
				"patchline" : { "source" : [ "obj-notein", 0 ], "destination" : [ "obj-pack", 0 ] }
			},
			{
				"patchline" : { "source" : [ "obj-notein", 1 ], "destination" : [ "obj-pack", 1 ] }
			},
			{
				"patchline" : { "source" : [ "obj-pack", 0 ], "destination" : [ "obj-prepend-note", 0 ] }
			},
			{
				"patchline" : { "source" : [ "obj-prepend-note", 0 ], "destination" : [ "obj-udpsend", 0 ] }
			},
			{
				"patchline" : { "source" : [ "obj-udprecv", 0 ], "destination" : [ "obj-route", 0 ] }
			},
			{
				"patchline" : { "source" : [ "obj-route", 0 ], "destination" : [ "obj-unpack-on", 0 ] }
			},
			{
				"patchline" : { "source" : [ "obj-route", 1 ], "destination" : [ "obj-unpack-off", 0 ] }
			},
			{
				"patchline" : { "source" : [ "obj-route", 2 ], "destination" : [ "obj-print-status", 0 ] }
			},
			{
				"patchline" : { "source" : [ "obj-unpack-on", 0 ], "destination" : [ "obj-noteout-on", 0 ] }
			},
			{
				"patchline" : { "source" : [ "obj-unpack-on", 1 ], "destination" : [ "obj-noteout-on", 1 ] }
			},
			{
				"patchline" : { "source" : [ "obj-unpack-on", 2 ], "destination" : [ "obj-noteout-on", 2 ] }
			},
			{
				"patchline" : { "source" : [ "obj-unpack-off", 0 ], "destination" : [ "obj-noteout-off", 0 ] }
			},
			{
				"patchline" : { "source" : [ "obj-int-off-vel", 0 ], "destination" : [ "obj-noteout-off", 1 ] }
			},
			{
				"patchline" : { "source" : [ "obj-unpack-off", 1 ], "destination" : [ "obj-noteout-off", 2 ] }
			},
			{
				"patchline" : { "source" : [ "obj-toggle", 0 ], "destination" : [ "obj-sel", 0 ] }
			},
			{
				"patchline" : { "source" : [ "obj-sel", 0 ], "destination" : [ "obj-msg-stop", 0 ] }
			},
			{
				"patchline" : { "source" : [ "obj-sel", 1 ], "destination" : [ "obj-msg-start", 0 ] }
			},
			{
				"patchline" : { "source" : [ "obj-msg-stop", 0 ], "destination" : [ "obj-udpsend", 0 ] }
			},
			{
				"patchline" : { "source" : [ "obj-msg-start", 0 ], "destination" : [ "obj-udpsend", 0 ] }
			},
			{
				"patchline" : { "source" : [ "obj-flonum-win", 0 ], "destination" : [ "obj-prepend-win", 0 ] }
			},
			{
				"patchline" : { "source" : [ "obj-prepend-win", 0 ], "destination" : [ "obj-udpsend", 0 ] }
			},
			{
				"patchline" : { "source" : [ "obj-flonum-topp", 0 ], "destination" : [ "obj-prepend-topp", 0 ] }
			},
			{
				"patchline" : { "source" : [ "obj-prepend-topp", 0 ], "destination" : [ "obj-udpsend", 0 ] }
			},
			{
				"patchline" : { "source" : [ "obj-flonum-temp", 0 ], "destination" : [ "obj-prepend-temp", 0 ] }
			},
			{
				"patchline" : { "source" : [ "obj-prepend-temp", 0 ], "destination" : [ "obj-udpsend", 0 ] }
			}
		],
		"parameters" : {},
		"dependency_cache" : [],
		"autosave" : 0
	}
}
