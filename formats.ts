// put this in pokemon-showdown/config/

export const Formats: import('../sim/dex-formats').FormatList = [
	{
		section: "S/V Doubles",
	},
	{
		name: "[Gen 9] VGC 2026 All Regs", // custom ruleset that allows all teams
		mod: 'gen9',
		gameType: 'doubles',
		bestOfDefault: true,
		ruleset: ['Flat Rules', '!! Adjust Level = 50', 'Min Source Gen = 9', 'VGC Timer', 'Open Team Sheets', 'Limit Two Restricted'],
		restricted: ['Restricted Legendary', 'Mythical'],
	},
];
